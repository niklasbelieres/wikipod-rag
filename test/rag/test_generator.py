from unittest.mock import MagicMock, patch

import pytest

from wikipod.config import LLMConfig
from wikipod.rag.generator import Generator


def test_generate_ollama_returns_content_from_response():
    config = LLMConfig(backend="ollama", ollama_model="qwen2.5:1.5b")
    generator = Generator(config)
    
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "message": {
            "content": "echo hello"
        }
    }
    
    mock_response.raise_for_status.return_value = None
    
    with patch("wikipod.rag.generator.requests.post", return_value=mock_response) as mock_post:
        result = generator.generate([{
            "role": "user",
            "content": "hello"
        }])
        
        assert result == "echo hello"
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == f"{config.ollama_host}/api/chat"
        assert kwargs["json"]["model"] == "qwen2.5:1.5b"
        assert kwargs["json"]["messages"] == [{"role": "user", "content": "hello"}]
        
def test_generate_llama_cpp_retruns_content_from_response():
    config = LLMConfig(backend="llama_cpp", model_path="fake/model.gguf")
    generator = Generator(config)
    
    mock_model = MagicMock()
    mock_model.create_chat_completion.return_value = {
        "choices": [{
            "message": {
                "content": "answer text"
            }
        }]
    }
    
    with patch("wikipod.rag.generator._load_llama_cpp_model", return_value=mock_model):
        result = generator.generate([{
            "role": "user",
            "content": "hello"
        }])
        
        assert result == "answer text"
        mock_model.create_chat_completion.assert_called_once()

def test_generate_raises_for_unknown_backend():
    generator = Generator(LLMConfig(backend="something_else"))
    with pytest.raises(ValueError):
        generator.generate([])


def test_generate_ollama_requires_ollama_model_configured():
    generator = Generator(LLMConfig(backend="ollama", ollama_model=None))
    with pytest.raises(ValueError):
        generator.generate([])