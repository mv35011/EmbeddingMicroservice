# Memory-friendly tester for 512MB RAM limit
import requests
import json
import time
from typing import List, Dict, Any


class MemoryFriendlyTester:
    def __init__(self, base_url: str = "https://embeddingmicroservice-5.onrender.com"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()

    def test_health_check(self) -> Dict[str, Any]:
        """Test the health endpoint"""
        print("🔍 Testing health check...")
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=30)
            response.raise_for_status()
            result = response.json()
            print(f"✅ Health check passed: {result}")
            return result
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return {"error": str(e)}

    def test_single_embedding(self, text: str = "Hello, world!") -> Dict[str, Any]:
        """Test single text embedding"""
        print(f"🔍 Testing single embedding for: '{text}'")
        try:
            payload = {"text": text}
            response = self.session.post(
                f"{self.base_url}/embed_single",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()

            embedding = result.get("embedding", [])
            print(f"✅ Single embedding success!")
            print(f"   - Dimension: {len(embedding)}")
            print(f"   - Model: {result.get('embedding_model_name')}")
            print(f"   - First 5 values: {embedding[:5]}")
            return result
        except Exception as e:
            print(f"❌ Single embedding failed: {e}")
            return {"error": str(e)}

    def test_small_batch(self, max_texts: int = 5) -> Dict[str, Any]:
        """Test small batch embedding (memory-friendly)"""
        texts = [
                    "The quick brown fox jumps over the lazy dog",
                    "Machine learning is transforming technology",
                    "Python is a versatile programming language"
                ][:max_texts]  # Limit to max_texts

        print(f"🔍 Testing small batch ({len(texts)} texts)...")
        try:
            payload = {"texts": texts}
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/embed",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            end_time = time.time()

            result = response.json()
            embeddings = result.get("embeddings", [])

            print(f"✅ Small batch success!")
            print(f"   - Processing time: {end_time - start_time:.2f}s")
            print(f"   - Number of embeddings: {len(embeddings)}")
            print(f"   - Dimension: {result.get('dimension')}")
            return result
        except Exception as e:
            print(f"❌ Small batch failed: {e}")
            return {"error": str(e)}

    def test_batch_size_limits(self):
        """Test different batch sizes to find the limit"""
        print("🔍 Testing batch size limits...")

        for batch_size in [1, 3, 5, 10, 15, 20, 25]:
            try:
                texts = [f"Test sentence number {i}" for i in range(batch_size)]
                payload = {"texts": texts}

                response = self.session.post(
                    f"{self.base_url}/embed",
                    json=payload,
                    timeout=60
                )

                if response.status_code == 200:
                    print(f"   ✅ Batch size {batch_size}: SUCCESS")
                else:
                    print(f"   ❌ Batch size {batch_size}: FAILED ({response.status_code})")
                    break

            except Exception as e:
                print(f"   ❌ Batch size {batch_size}: ERROR - {e}")
                break

    def test_sequential_processing(self, total_texts: int = 20):
        """Process large batches sequentially"""
        print(f"🔍 Testing sequential processing of {total_texts} texts...")

        all_texts = [f"Sequential test sentence number {i}" for i in range(total_texts)]
        batch_size = 5  # Process in small batches
        all_embeddings = []

        try:
            for i in range(0, len(all_texts), batch_size):
                batch = all_texts[i:i + batch_size]
                print(f"   Processing batch {i // batch_size + 1}/{(len(all_texts) - 1) // batch_size + 1}")

                payload = {"texts": batch}
                response = self.session.post(
                    f"{self.base_url}/embed",
                    json=payload,
                    timeout=30
                )
                response.raise_for_status()

                result = response.json()
                all_embeddings.extend(result["embeddings"])

                # Small delay to be nice to the server
                time.sleep(0.5)

            print(f"✅ Sequential processing complete!")
            print(f"   - Total embeddings: {len(all_embeddings)}")
            return all_embeddings

        except Exception as e:
            print(f"❌ Sequential processing failed: {e}")
            return []

    def run_memory_friendly_tests(self):
        """Run memory-optimized tests"""
        print("🚀 Starting memory-friendly tests...\n")

        # Basic tests
        health_result = self.test_health_check()
        print()

        if health_result.get("embedding_model_loaded"):
            self.test_single_embedding()
            print()

            self.test_small_batch(max_texts=3)
            print()

            self.test_batch_size_limits()
            print()

            self.test_sequential_processing(total_texts=15)
            print()
        else:
            print("❌ Model not loaded, skipping embedding tests")

        print("🎉 Memory-friendly tests completed!")


# Updated LangChain integration with batching
class MemoryFriendlyEmbeddingService:
    """Memory-friendly LangChain-compatible embedding class"""

    def __init__(self, service_url: str = "https://embeddingmicroservice-5.onrender.com", batch_size: int = 5):
        self.service_url = service_url.rstrip('/')
        self.session = requests.Session()
        self.batch_size = batch_size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents in small batches"""
        if len(texts) <= self.batch_size:
            # Small batch - process normally
            try:
                payload = {"texts": texts}
                response = self.session.post(
                    f"{self.service_url}/embed",
                    json=payload,
                    timeout=60
                )
                response.raise_for_status()
                result = response.json()
                return result["embeddings"]
            except Exception as e:
                raise Exception(f"Failed to embed documents: {e}")
        else:
            # Large batch - process sequentially
            all_embeddings = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                try:
                    payload = {"texts": batch}
                    response = self.session.post(
                        f"{self.service_url}/embed",
                        json=payload,
                        timeout=60
                    )
                    response.raise_for_status()
                    result = response.json()
                    all_embeddings.extend(result["embeddings"])
                    time.sleep(0.2)  # Small delay between batches
                except Exception as e:
                    raise Exception(f"Failed to embed document batch {i // self.batch_size + 1}: {e}")
            return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text"""
        try:
            payload = {"text": text}
            response = self.session.post(
                f"{self.service_url}/embed_single",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return result["embedding"]
        except Exception as e:
            raise Exception(f"Failed to embed query: {e}")


def test_memory_friendly_langchain():
    """Test memory-friendly LangChain integration"""
    print("🔍 Testing memory-friendly LangChain integration...")

    embeddings = MemoryFriendlyEmbeddingService(batch_size=3)

    try:
        # Test with small document set first
        documents = [
            "LangChain framework for language models",
            "Vector databases for similarity search",
            "Text embeddings numerical representations"
        ]

        print("   - Embedding small document set...")
        doc_embeddings = embeddings.embed_documents(documents)
        print(f"   - Got {len(doc_embeddings)} embeddings")

        # Test query
        query = "What is LangChain?"
        query_embedding = embeddings.embed_query(query)
        print(f"   - Query embedding dimension: {len(query_embedding)}")

        # Test larger document set with batching
        larger_docs = [f"Document {i} about machine learning and AI" for i in range(12)]
        print(f"   - Embedding {len(larger_docs)} documents with batching...")
        larger_embeddings = embeddings.embed_documents(larger_docs)
        print(f"   - Got {len(larger_embeddings)} embeddings")

        print("✅ Memory-friendly LangChain integration passed!")
        return True

    except Exception as e:
        print(f"❌ Memory-friendly integration failed: {e}")
        return False


if __name__ == "__main__":
    tester = MemoryFriendlyTester()
    tester.run_memory_friendly_tests()

    print("\n" + "=" * 50 + "\n")

    test_memory_friendly_langchain()