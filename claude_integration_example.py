#!/usr/bin/env python3
"""
Example implementation showing how to integrate Anthropic's Claude API
with the YouTube Content RAG system for enhanced answer generation.
"""

import os
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Uncomment to use Anthropic Claude
# import anthropic

from rag_system import RAGSystem

class ClaudeRAGSystem(RAGSystem):
    """Enhanced RAG system using Anthropic's Claude for answer generation"""
    
    def __init__(self, collection_name: str = "youtube_transcripts"):
        super().__init__(collection_name)
        
        # Initialize Anthropic client (requires anthropic>=0.25.0)
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        self.anthropic_client = None
        
        if self.anthropic_api_key:
            try:
                # import anthropic
                # self.anthropic_client = anthropic.Anthropic(
                #     api_key=self.anthropic_api_key
                # )
                print("✅ Anthropic Claude configured (commented out for demo)")
            except ImportError:
                print("⚠️  Anthropic library not installed. Run: pip install anthropic>=0.25.0")
        else:
            print("⚠️  ANTHROPIC_API_KEY not found in environment")
    
    def generate_answer_with_claude(self, question: str, top_k: int = 5) -> Dict:
        """
        Generate enhanced answers using Claude with proper context management
        """
        try:
            # Step 1: Retrieve relevant context
            search_results = self.search(question, top_k)
            
            if not search_results:
                return {
                    'answer': "I couldn't find any relevant information in the video transcripts.",
                    'sources': [],
                    'model_used': 'fallback'
                }
            
            # Step 2: Prepare context with source tracking
            context_parts = []
            sources = []
            
            for i, result in enumerate(search_results):
                video_title = result['metadata'].get('title', 'Unknown Video')
                video_url = result['metadata'].get('url', '')
                text = result['text']
                similarity = result['similarity_score']
                
                # Add numbered source reference
                source_ref = f"[{i+1}]"
                context_parts.append(f"{source_ref} {text}")
                
                sources.append({
                    'reference': source_ref,
                    'title': video_title,
                    'url': video_url,
                    'similarity_score': similarity,
                    'text_preview': text[:100] + "..." if len(text) > 100 else text
                })
            
            context = "\n\n".join(context_parts)
            
            # Step 3: Generate answer with Claude (if available)
            if self.anthropic_client:
                answer = self._generate_with_claude(question, context, sources)
                model_used = 'claude-3-sonnet'
            else:
                # Fallback to structured context display
                answer = self._generate_fallback_answer(question, context, sources)
                model_used = 'fallback'
            
            return {
                'answer': answer,
                'sources': sources,
                'model_used': model_used,
                'context_chunks': len(search_results)
            }
            
        except Exception as e:
            print(f"Error generating answer: {str(e)}")
            return {
                'answer': f"An error occurred while generating the answer: {str(e)}",
                'sources': [],
                'model_used': 'error'
            }
    
    def _generate_with_claude(self, question: str, context: str, sources: List[Dict]) -> str:
        """Generate answer using Claude API"""
        
        # System prompt for Claude
        system_prompt = """You are an expert assistant that answers questions based on YouTube video transcripts. 

Your task is to:
1. Analyze the provided transcript excerpts carefully
2. Provide accurate, helpful answers based on the context
3. Use source references [1], [2], etc. when citing specific information
4. If the transcripts don't fully answer the question, acknowledge the limitations
5. Keep your response clear, concise, and well-structured

Always cite your sources using the provided reference numbers."""

        # User prompt with context and question
        user_prompt = f"""Based on the following transcript excerpts from YouTube videos, please answer the question below.

TRANSCRIPT EXCERPTS:
{context}

QUESTION: {question}

Please provide a comprehensive answer based on the transcript content above, using source references [1], [2], etc. when citing specific information."""

        try:
            # Generate with Claude (commented out for demo)
            # response = self.anthropic_client.messages.create(
            #     model="claude-3-sonnet-20240229",
            #     max_tokens=1000,
            #     temperature=0.7,
            #     system=system_prompt,
            #     messages=[{
            #         "role": "user",
            #         "content": user_prompt
            #     }]
            # )
            # return response.content[0].text
            
            # Demo response (replace with actual Claude call)
            return f"""Based on the provided video transcripts, here's what I found regarding your question:

{question}

The transcript excerpts contain relevant information from {len(sources)} video sources. [1] provides key insights about the main topic, while [2] and [3] offer additional context and examples.

To get the actual Claude-generated response, please:
1. Install anthropic: pip install anthropic>=0.25.0
2. Set your ANTHROPIC_API_KEY environment variable
3. Uncomment the Claude API calls in this file

The current implementation shows the structure but uses a demo response."""
            
        except Exception as e:
            return f"Error calling Claude API: {str(e)}"
    
    def _generate_fallback_answer(self, question: str, context: str, sources: List[Dict]) -> str:
        """Generate structured answer without LLM"""
        
        answer_parts = [
            f"Based on the video transcripts, here are the most relevant excerpts for your question:",
            f"",
            f"**Question:** {question}",
            f"",
            f"**Relevant Information:**"
        ]
        
        for source in sources:
            answer_parts.extend([
                f"",
                f"{source['reference']} **{source['title']}**",
                f"   {source['text_preview']}",
                f"   *Relevance: {source['similarity_score']:.3f}*",
                f"   *Video: {source['url']}*"
            ])
        
        answer_parts.extend([
            f"",
            f"💡 **To get AI-generated answers:**",
            f"   • Add your ANTHROPIC_API_KEY to .env file",
            f"   • Install: pip install anthropic>=0.25.0",
            f"   • Uncomment Claude integration code"
        ])
        
        return "\n".join(answer_parts)

def demo_claude_integration():
    """Demonstrate the Claude integration capabilities"""
    
    print("🤖 Claude Integration Demo")
    print("=" * 50)
    
    # Initialize enhanced RAG system
    claude_rag = ClaudeRAGSystem()
    
    # Check if we have existing data
    stats = claude_rag.get_stats()
    if stats.get('total_chunks', 0) == 0:
        print("⚠️  No transcript data found in database.")
        print("   Run the main application first to extract some video transcripts.")
        return
    
    print(f"📊 Database contains {stats['total_chunks']} transcript chunks")
    
    # Test questions
    test_questions = [
        "What are the main topics covered in these videos?",
        "Can you explain the key concepts discussed?",
        "What practical advice is given in the content?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n🔍 Test Question {i}: {question}")
        print("-" * 40)
        
        # Generate answer with Claude integration
        result = claude_rag.generate_answer_with_claude(question, top_k=3)
        
        print(f"**Answer ({result['model_used']}):**")
        print(result['answer'])
        
        if result['sources']:
            print(f"\n**Sources ({len(result['sources'])} found):**")
            for source in result['sources'][:2]:  # Show first 2 sources
                print(f"  {source['reference']} {source['title']} (score: {source['similarity_score']:.3f})")
        
        print("\n" + "="*50)

if __name__ == "__main__":
    load_dotenv()
    demo_claude_integration()