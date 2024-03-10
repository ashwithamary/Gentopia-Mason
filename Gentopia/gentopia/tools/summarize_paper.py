import aiohttp
import asyncio
import os
from typing import Any, Optional, Type, List
import fitz  # PyMuPDF for handling PDF files
from scholarly import scholarly
from pydantic import BaseModel, Field
from gentopia.tools.basetool import BaseTool
import textwrap

# Load OpenAI API key from environment variables or default to '<your-api-key>'
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", '<your-api-key>')

# Define arguments for the tool using Pydantic models
class SummarizePaperArgs(BaseModel):
    query: Optional[str] = Field(None, description="Search query to find the paper or direct URL")

# The main class for summarizing papers
class SummarizePaper(BaseTool):
    name = "summarize_paper"
    description = "Finds and summarizes a paper based on a search query or direct URL"
    args_schema: Optional[Type[BaseModel]] = SummarizePaperArgs

    # Asynchronous run method that orchestrates the summarization process
    async def _arun(self, query: Optional[str] = None, pdf_url: Optional[str] = None) -> str:
        # If no direct PDF URL is provided, try to get it using the query
        if not pdf_url and query:
            pdf_url = await asyncio.to_thread(self.get_pdf_url, query)

        # If we can't get a URL, return an error message
        if not pdf_url:
            return "Could not find a URL for the paper."
        
        # Download and extract text from the PDF, then summarize it
        extracted_text = await self.download_and_extract_text(pdf_url)
        summary = await self.summarize_text_with_openai(extracted_text)
        return summary
    
    # Synchronous method to get the PDF URL from a query using Google Scholar
    def get_pdf_url(self, title):
        paper_search_result = scholarly.search_single_pub(title)
        return paper_search_result.get('eprint_url') if paper_search_result else None

    # Asynchronously download the paper and extract text from it
    async def download_and_extract_text(self, url: str) -> str:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    pdf_path = '/tmp/temp_paper.pdf'
                    content = await response.read()
                    with open(pdf_path, 'wb') as f:
                        f.write(content)
                    
                    document = fitz.open(pdf_path)
                    text = ""
                    for page in document:
                        text += page.get_text()
                    document.close()
                    os.remove(pdf_path)
                    return text
                else:
                    return "Failed to download the PDF from the provided URL."

    # Asynchronously summarize the extracted text using OpenAI
    async def summarize_text_with_openai(self, text: str) -> str:
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        chunk_size = 3500  
        chunks = textwrap.wrap(text, chunk_size)

        batch_size = 5  
        delay_between_batches = 10  

        batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
        
        summaries = []  

        for batch in batches:
            tasks = [self.summarize_chunk(chunk, headers) for chunk in batch]
            batch_summaries = await asyncio.gather(*tasks)
            summaries.extend(batch_summaries)
            await asyncio.sleep(delay_between_batches)  
        
        final_summary = ' '.join(summaries)
        return final_summary

    async def summarize_chunk(self, chunk: str, headers: dict) -> str:
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Summarize this text: {chunk}"},
            ],
            "max_tokens": 250,
        }
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.post("https://api.openai.com/v1/chat/completions", json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['choices'][0]['message']['content']
                else:
                    response_text = await response.text()
                    print(f"OpenAI API call failed for a chunk with status {response.status}: {response_text}")
                    return "OpenAI API request failed for a chunk."

    # This tool is designed to be run asynchronously and doesn't support synchronous execution
    def _run(self, *args: Any, **kwargs: Any) -> Any:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            raise NotImplementedError("Synchronous call in a running event loop not supported.")
        else:
            result = loop.run_until_complete(self._arun(*args, **kwargs))
            return result