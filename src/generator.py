import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.config import Config

class LLMGenerator:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self, model_name: str = Config.LLM_MODEL_NAME):
        """
        Loads the Qwen model. Uses 4-bit quantization if CUDA (GPU) is available
        to optimize memory usage (essential for Kaggle T4 GPUs).
        """
        print(f"Loading LLM generator model: {model_name} on device: {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if self.device == "cuda":
            # 4-bit quantization config to run efficiently on T4 GPU
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto"
            )
        else:
            print("WARNING: CUDA is not available. Loading LLM on CPU. This will be slow and require substantial system RAM.")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="cpu",
                torch_dtype=torch.float32
            )
            
        print("LLM generator model loaded successfully.")

    def generate(self, query: str, retrieved_docs: list, corpus: dict) -> str:
        """
        Formats a prompt with the retrieved documents as context and generates the answer.
        
        Args:
            query: The user's search query.
            retrieved_docs: A list of retrieved documents: [{'corpus-id': doc_id, 'score': score}]
            corpus: A dictionary mapping 'corpus-id' to the raw document text.
            
        Returns:
            The generated Vietnamese answer from the LLM.
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("LLM model is not loaded. Call load_model() first.")
            
        # Format the context from retrieved documents
        context_parts = []
        for i, item in enumerate(retrieved_docs):
            doc_id = item["corpus-id"]
            doc_text = corpus.get(doc_id, "")
            context_parts.append(f"[Tài liệu tham khảo {i+1}]:\n{doc_text}\n")
            
        context = "\n".join(context_parts)
        
        # Build chat prompt in Vietnamese
        messages = [
            {
                "role": "system",
                "content": (
                    "Bạn là một trợ lý AI chuyên nghiệp của GreenNode. Hãy trả lời câu hỏi của người dùng "
                    "một cách ngắn gọn, chính xác dựa trên thông tin bảng và dữ liệu ngữ cảnh được cung cấp bên dưới. "
                    "Nếu ngữ cảnh không có thông tin để trả lời, hãy nói 'Tôi không tìm thấy thông tin phù hợp trong dữ liệu'."
                )
            },
            {
                "role": "user",
                "content": f"Dữ liệu ngữ cảnh tham khảo:\n{context}\n\nCâu hỏi: {query}"
            }
        ]
        
        # Format the prompt using model's chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize prompt and send to device
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        
        # Generate answer
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=Config.LLM_MAX_NEW_TOKENS,
                temperature=Config.LLM_TEMPERATURE,
                do_sample=False, # Greedier decoding for factual answering
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        # Extract the generated response part (exclude prompt)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response.strip()
