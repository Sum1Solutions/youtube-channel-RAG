import chromadb
from sentence_transformers import SentenceTransformer
import os
from typing import List, Dict, Optional
import json
import uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai
from dotenv import load_dotenv

load_dotenv()

class RAGSystem:
    def __init__(self, collection_name: str = "youtube_transcripts"):
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection_name = collection_name
        
        # Initialize sentence transformer for embeddings
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Initialize OpenAI client (optional, for answer generation)
        openai.api_key = os.getenv('OPENAI_API_KEY')
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "YouTube video transcripts for RAG"}
            )
            print(f"Created new collection: {collection_name}")
    
    def chunk_transcript(self, transcript: str, video_info: Dict) -> List[Dict]:
        """Split transcript into chunks for better retrieval"""
        chunks = self.text_splitter.split_text(transcript)
        
        chunked_docs = []
        for i, chunk in enumerate(chunks):
            chunked_docs.append({
                'id': f"{video_info['video_id']}_chunk_{i}",
                'text': chunk,
                'metadata': {
                    'video_id': video_info['video_id'],
                    'title': video_info['title'],
                    'url': video_info['url'],
                    'chunk_index': i,
                    'upload_date': video_info.get('upload_date', 'unknown'),
                    'duration': str(video_info.get('duration', 'unknown'))
                }
            })
        
        return chunked_docs
    
    def add_transcripts(self, transcripts: List[Dict]):
        """Add transcripts to the vector database"""
        all_chunks = []
        
        for transcript_data in transcripts:
            chunks = self.chunk_transcript(transcript_data['transcript'], transcript_data)
            all_chunks.extend(chunks)
        
        if not all_chunks:
            print("No chunks to add to database")
            return
        
        # Prepare data for ChromaDB
        ids = [chunk['id'] for chunk in all_chunks]
        documents = [chunk['text'] for chunk in all_chunks]
        metadatas = [chunk['metadata'] for chunk in all_chunks]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(documents).tolist()
        
        # Add to ChromaDB
        try:
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            print(f"Added {len(all_chunks)} chunks to the database")
        except Exception as e:
            print(f"Error adding to database: {str(e)}")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for relevant transcript chunks"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()[0]
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity_score': 1 - results['distances'][0][i],  # Convert distance to similarity
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error searching: {str(e)}")
            return []
    
    def generate_answer(self, question: str, top_k: int = 3) -> str:
        """Generate an answer using retrieved context and OpenAI"""
        try:
            # Get relevant context
            search_results = self.search(question, top_k)
            
            if not search_results:
                return "I couldn't find any relevant information in the video transcripts."
            
            # Prepare context
            context_parts = []
            for result in search_results:
                video_title = result['metadata'].get('title', 'Unknown Video')
                video_url = result['metadata'].get('url', '')
                text = result['text']
                
                context_parts.append(f"From video '{video_title}' ({video_url}):\n{text}")
            
            context = "\n\n".join(context_parts)
            
            # Check if OpenAI API key is available
            if not openai.api_key:
                # Return context without AI generation
                return f"Based on the video transcripts:\n\n{context}"
            
            # Generate answer using OpenAI
            prompt = f"""Based on the following transcript excerpts from YouTube videos, please answer the question.

Context from video transcripts:
{context}

Question: {question}

Please provide a comprehensive answer based on the transcript content above. If the transcripts don't contain enough information to fully answer the question, mention that limitation.

Answer:"""

            client = openai.OpenAI(api_key=openai.api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on YouTube video transcripts. Provide accurate, helpful answers based on the provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating answer: {str(e)}")
            # Fallback to basic search results
            search_results = self.search(question, top_k)
            if search_results:
                return f"Here are the most relevant excerpts I found:\n\n" + \
                       "\n\n".join([f"From '{result['metadata']['title']}':\n{result['text']}" 
                                   for result in search_results[:2]])
            else:
                return "I couldn't find any relevant information in the video transcripts."
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        try:
            count = self.collection.count()
            return {
                'total_chunks': count,
                'collection_name': self.collection_name
            }
        except Exception as e:
            return {'error': str(e)}
    
    def clear_database(self):
        """Clear all data from the database"""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "YouTube video transcripts for RAG"}
            )
            print("Database cleared successfully")
        except Exception as e:
            print(f"Error clearing database: {str(e)}")