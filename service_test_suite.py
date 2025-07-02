import requests
import json
import time
from typing import List, Dict, Any


class EmbeddingServiceTester:
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

    def test_root_endpoint(self) -> Dict[str, Any]:
        """Test the root endpoint"""
        print("🔍 Testing root endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/", timeout=30)
            response.raise_for_status()
            result = response.json()
            print(f"✅ Root endpoint passed: {result}")
            return result
        except Exception as e:
            print(f"❌ Root endpoint failed: {e}")
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

    def test_batch_embeddings(self, texts: List[str] = None) -> Dict[str, Any]:
        """Test batch text embeddings"""
        if texts is None:
            texts = [
                "The quick brown fox jumps over the lazy dog",
                "Machine learning is transforming technology",
                "Python is a versatile programming language",
                "Natural language processing enables AI understanding"
            ]

        print(f"🔍 Testing batch embeddings for {len(texts)} texts...")
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

            print(f"✅ Batch embedding success!")
            print(f"   - Processing time: {end_time - start_time:.2f}s")
            print(f"   - Number of embeddings: {len(embeddings)}")
            print(f"   - Dimension: {result.get('dimension')}")
            print(f"   - Model: {result.get('embedding_model_name')}")

            return result
        except Exception as e:
            print(f"❌ Batch embedding failed: {e}")
            return {"error": str(e)}

    def test_empty_input(self):
        """Test error handling with empty input"""
        print("🔍 Testing empty input handling...")
        try:
            payload = {"texts": []}
            response = self.session.post(f"{self.base_url}/embed", json=payload)
            print(f"   - Status code: {response.status_code}")
            print(f"   - Response: {response.json()}")
        except Exception as e:
            print(f"   - Exception: {e}")

    def test_large_text(self):
        """Test with large text input"""
        print("🔍 Testing large text input...")
        large_text = "This is a test sentence. " * 100  # ~600 words
        try:
            result = self.test_single_embedding(large_text)
            if "error" not in result:
                print("✅ Large text handled successfully")
            else:
                print("❌ Large text failed")
        except Exception as e:
            print(f"❌ Large text test failed: {e}")

    def similarity_test(self):
        """Test semantic similarity"""
        print("🔍 Testing semantic similarity...")
        texts = [
            "The cat sits on the mat",
            "A feline rests on the rug",  # Similar meaning
            "Python programming language",  # Different meaning
        ]

        try:
            result = self.test_batch_embeddings(texts)
            if "error" not in result:
                embeddings = result["embeddings"]

                # Calculate cosine similarities
                import numpy as np
                def cosine_similarity(a, b):
                    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

                sim_1_2 = cosine_similarity(embeddings[0], embeddings[1])
                sim_1_3 = cosine_similarity(embeddings[0], embeddings[2])

                print(f"   - Similarity (cat/mat vs feline/rug): {sim_1_2:.3f}")
                print(f"   - Similarity (cat/mat vs python): {sim_1_3:.3f}")

                if sim_1_2 > sim_1_3:
                    print("✅ Semantic similarity working correctly!")
                else:
                    print("⚠️ Unexpected similarity results")
        except ImportError:
            print("   - Skipping similarity calculation (numpy not available)")
        except Exception as e:
            print(f"❌ Similarity test failed: {e}")

    def run_all_tests(self):
        """Run all tests"""
        print("🚀 Starting comprehensive embedding service tests...\n")

        # Basic functionality tests
        self.test_root_endpoint()
        print()

        health_result = self.test_health_check()
        print()

        if health_result.get("embedding_model_loaded"):
            self.test_single_embedding()
            print()

            self.test_batch_embeddings()
            print()

            self.test_empty_input()
            print()

            self.test_large_text()
            print()

            self.similarity_test()
            print()
        else:
            print("❌ Model not loaded, skipping embedding tests")

        print("🎉 All tests completed!")


# LangChain Custom Embedding Class
from typing import List
import requests


class CustomEmbeddingService:
    """Custom LangChain-compatible embedding class for your service"""

    def __init__(self, service_url: str = "https://embeddingmicroservice-5.onrender.com"):
        self.service_url = service_url.rstrip('/')
        self.session = requests.Session()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
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


# Example usage with LangChain
def test_langchain_integration():
    """Test the custom embedding service with LangChain-style usage"""
    print("🔍 Testing LangChain integration...")

    # Initialize custom embedding service
    embeddings = CustomEmbeddingService()

    try:
        # Test document embedding
        documents = [
            "LangChain is a framework for developing applications powered by language models.",
            "Vector databases store high-dimensional vectors for similarity search.",
            "Embeddings convert text into numerical representations."
        ]

        print("   - Embedding documents...")
        doc_embeddings = embeddings.embed_documents(documents)
        print(f"   - Got {len(doc_embeddings)} document embeddings")

        # Test query embedding
        query = "What is LangChain used for?"
        print("   - Embedding query...")
        query_embedding = embeddings.embed_query(query)
        print(f"   - Got query embedding with dimension {len(query_embedding)}")

        print("✅ LangChain integration test passed!")
        return True

    except Exception as e:
        print(f"❌ LangChain integration failed: {e}")
        return False


# Vector similarity search example
def vector_search_example():
    """Example of using embeddings for vector similarity search"""
    print("🔍 Testing vector similarity search...")

    embeddings = CustomEmbeddingService()

    try:
        # Sample knowledge base
        documents = [
            "Python is a high-level programming language",
            "Machine learning algorithms learn patterns from data",
            "Natural language processing analyzes human language",
            "Deep learning uses neural networks with multiple layers",
            "Data science combines statistics and programming"
        ]

        # Get embeddings for all documents
        doc_embeddings = embeddings.embed_documents(documents)

        # Query
        query = "What is machine learning?"
        query_embedding = embeddings.embed_query(query)

        # Calculate similarities (using dot product as approximation)
        similarities = []
        for i, doc_emb in enumerate(doc_embeddings):
            # Simple dot product similarity
            similarity = sum(a * b for a, b in zip(query_embedding, doc_emb))
            similarities.append((similarity, documents[i]))

        # Sort by similarity
        similarities.sort(reverse=True)

        print("   - Top 3 most similar documents:")
        for i, (score, doc) in enumerate(similarities[:3], 1):
            print(f"   {i}. (Score: {score:.3f}) {doc}")

        print("✅ Vector search example completed!")

    except Exception as e:
        print(f"❌ Vector search example failed: {e}")


if __name__ == "__main__":
    # Run basic tests
    tester = EmbeddingServiceTester()
    tester.run_all_tests()

    print("\n" + "=" * 50 + "\n")

    # Test LangChain integration
    test_langchain_integration()

    print("\n" + "=" * 50 + "\n")

    # Test vector search
    vector_search_example()