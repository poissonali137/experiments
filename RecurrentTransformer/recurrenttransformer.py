from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import torch.nn as nn
import bitsandbytes as bnb

class RecurrentProver(nn.Module):
    def __init__(self):
        super().__init__()
        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/deepseek-math-7b-instruct",
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "deepseek-ai/deepseek-math-7b-instruct"
        )
        self.layer_norm = nn.LayerNorm(self.model.config.hidden_size)
        
    def forward(self, prompt, num_passes=3, temperature=0.7):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        hidden_state = self.model.get_input_embeddings()(inputs.input_ids)
        
        # Process through multiple passes
        for i in range(num_passes - 1):  # -1 because last pass is for generation
            combined = hidden_state + self.model.get_input_embeddings()(inputs.input_ids)
            
            outputs = self.model(
                inputs_embeds=combined,
                output_hidden_states=True
            )
            
            hidden_state = self.layer_norm(outputs.hidden_states[-1])
        
        # Final pass with generation
        combined = hidden_state + self.model.get_input_embeddings()(inputs.input_ids)
        outputs = self.model.generate(
            inputs.input_ids,
            inputs_embeds=combined,  # Use our refined embeddings
            max_length=2048,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            use_cache=True
        )
        
        return self.tokenizer.decode(outputs[0])

# Usage example
if __name__ == "__main__":
    prover = RecurrentProver()
    
    math_problem = """Prove that for any positive integers a and b, 
    if a divides b and b divides a, then a = b."""
    
    refined_proof = prover(math_problem)
    print(refined_proof)