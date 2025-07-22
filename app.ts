import { Hono } from 'hono'
import { logger } from 'hono/logger'
import { promises as fs } from 'fs'
import { join } from 'path'
import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import {
  RecursiveCharacterTextSplitter
} from "@langchain/textsplitters"
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { Document } from "@langchain/core/documents";


import { ChatOpenAI } from "@langchain/openai";
import {pull} from "langchain/hub";
import {ChatPromptTemplate} from "@langchain/core/prompts";

const llm = new ChatOpenAI({
    model: "gpt-4o-mini",
    temperature: 0
});

const embeddings = new OpenAIEmbeddings({
    model: "text-embedding-3-large"
});
const vectorStore = new MemoryVectorStore(embeddings);



const app = new Hono()

app.use('*', logger())

async function loadDocumentFromFile(filePath: string): Promise<Document[]> {
  const fileExtension = filePath.split('.').pop()?.toLowerCase()
  
  switch (fileExtension) {
    case 'pdf':
      const pdfLoader = new PDFLoader(filePath)
      return await pdfLoader.load()
    case 'txt':
    case 'md':
    case 'markdown':
      const textContent = await fs.readFile(filePath, 'utf-8')
      return [new Document({ pageContent: textContent, metadata: { source: filePath } })]
    default:
      throw new Error(`Unsupported file type: ${fileExtension}`)
  }
}

async function splitAndStoreDocuments(documents: Document[]): Promise<number> {
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  })
  
  const splits = await textSplitter.splitDocuments(documents)
  
  await vectorStore.addDocuments(splits)
  
  return splits.length
}

async function generateMultipleQueries(originalQuestion: string): Promise<string[]> {
  const queryGenerationPrompt = ChatPromptTemplate.fromTemplate(`
You are an AI assistant that generates multiple search queries to help retrieve relevant information.
Given an original question, generate 3-4 alternative ways to ask the same question that might retrieve different but relevant information.

Original question: {question}

Generate alternative queries that:
1. Use different keywords or synonyms
2. Approach the topic from different angles
3. Are more specific or more general
4. Focus on different aspects of the question

Return only the alternative queries, one per line, without numbering or bullet points.
`);

  const messages = await queryGenerationPrompt.invoke({ question: originalQuestion });
  const response = await llm.invoke(messages);
  
  const queries = response.content.toString()
    .split('\n')
    .map(q => q.trim())
    .filter(q => q.length > 0);
  
  // Include the original question
  return [originalQuestion, ...queries];
}

async function retrieveWithMultipleQueries(queries: string[]): Promise<Document[]> {
  const allResults: Document[] = [];
  const seenContent = new Set<string>();
  
  for (const query of queries) {
    const results = await vectorStore.similaritySearch(query, 3);
    
    // Deduplicate based on content
    for (const doc of results) {
      if (!seenContent.has(doc.pageContent)) {
        seenContent.add(doc.pageContent);
        allResults.push(doc);
      }
    }
  }
  
  return allResults;
}

app.post('/upload', async (c) => {
  try {
    const body = await c.req.parseBody()
    const file = body['file'] as File
    
    if (!file) {
      return c.json({ error: 'No file provided' }, 400)
    }

    const publicDir = 'public'
    await fs.mkdir(publicDir, { recursive: true })
    
    const buffer = await file.arrayBuffer()
    const filePath = join(publicDir, file.name)
    await fs.writeFile(filePath, new Uint8Array(buffer))

    const documents = await loadDocumentFromFile(filePath)
    console.log(`Loaded ${documents.length} documents from ${file.name}`)
    
    const chunkCount = await splitAndStoreDocuments(documents)
    console.log(`Created ${chunkCount} chunks and stored in vector store`)
    
    return c.json({ 
      message: 'File processed successfully', 
      filename: file.name,
      documentsLoaded: documents.length,
      chunksCreated: chunkCount
    })
  } catch (error) {
    console.error('Error processing file:', error)
    return c.json({ 
      error: error instanceof Error ? error.message : 'Unknown error occurred' 
    }, 500)
  }
})

app.post('/chat', async (c) => {
  try {
    const body = await c.req.json()
    
    if (!body.question || typeof body.question !== 'string') {
      return c.json({ error: 'Question is required and must be a string' }, 400)
    }

    // Generate multiple queries for better retrieval
    console.log('Generating multiple queries for:', body.question)
    const queries = await generateMultipleQueries(body.question)
    console.log('Generated queries:', queries)

    // Retrieve documents using all queries
    const similarChunks = await retrieveWithMultipleQueries(queries)
    
    if (similarChunks.length === 0) {
      return c.json({ 
        answer: "I don't have any relevant information to answer your question. Please upload some documents first.",
        sources: [],
        queriesGenerated: queries
      })
    }

    // Limit to top chunks to avoid context overflow
    const topChunks = similarChunks.slice(0, 8)
    const context = topChunks.map(doc => doc.pageContent).join('\n\n')

    const promptTemplate = await pull<ChatPromptTemplate>("rlm/rag-prompt");

    const messages = await promptTemplate.invoke({question: body.question, context: context})
    const response = await llm.invoke(messages)

    return c.json({ 
      answer: response.content,
      sources: topChunks.map(doc => doc.metadata).filter(Boolean),
      chunksUsed: topChunks.length,
      queriesGenerated: queries,
      totalChunksRetrieved: similarChunks.length
    })
  } catch (error) {
    console.error('Error in chat:', error)
    return c.json({ 
      error: error instanceof Error ? error.message : 'Unknown error occurred' 
    }, 500)
  }
})

app.get('/status', async (c) => {
  try {
    const vectorStoreSize = vectorStore.memoryVectors.length
    return c.json({
      status: 'healthy',
      vectorStoreChunks: vectorStoreSize,
      supportedFileTypes: ['pdf', 'txt', 'md', 'markdown']
    })
  } catch (error) {
    return c.json({ 
      status: 'error',
      error: error instanceof Error ? error.message : 'Unknown error'
    }, 500)
  }
})

export default app
