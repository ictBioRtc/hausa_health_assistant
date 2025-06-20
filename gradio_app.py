import gradio as gr
import torch
import threading
import time
from datetime import datetime
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, standardize_data_formats, train_on_responses_only
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import random

# Set seed for reproducibility
random.seed(42)
torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Global variables
model = None
tokenizer = None
training_stats = {}
trainer = None
train_dataset = None
eval_dataset = None

# Thread-safe status tracking
class StatusManager:
    def __init__(self):
        self._lock = threading.Lock()
        self.status = "Ready"
        self.progress = 0
        self.model_loaded = False
        self.model_trained = False
        self.error = None
        
    def update_status(self, status: str, progress: int = None, error: str = None):
        with self._lock:
            self.status = status
            if progress is not None:
                self.progress = progress
            if error is not None:
                self.error = error
                
    def set_model_loaded(self, loaded: bool):
        with self._lock:
            self.model_loaded = loaded
            
    def set_model_trained(self, trained: bool):
        with self._lock:
            self.model_trained = trained
            
    def get_status(self):
        with self._lock:
            return {
                'status': self.status,
                'progress': self.progress,
                'model_loaded': self.model_loaded,
                'model_trained': self.model_trained,
                'error': self.error
            }

status_manager = StatusManager()

def initialize_model_background():
    """Initialize model in background thread"""
    global model, tokenizer
    
    try:
        status_manager.update_status("🔄 Loading base Gemma model...", 10)
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        status_manager.update_status("🔄 Downloading model weights...", 30)
        
        model, tokenizer = FastModel.from_pretrained(
            model_name="unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
            max_seq_length=2048,
            load_in_8bit=False,
            full_finetuning=False,
            load_in_4bit=False,
            trust_remote_code=True,
        )
        
        status_manager.update_status("🔄 Configuring chat template...", 70)
        
        # Configure chat template
        tokenizer = get_chat_template(
            tokenizer,
            chat_template="gemma-3",
        )
        
        status_manager.update_status("🔄 Moving to GPU...", 90)
        
        # Move to CUDA if available
        if torch.cuda.is_available():
            model = model.to("cuda")
            
        status_manager.update_status("✅ Model loaded successfully!", 100)
        status_manager.set_model_loaded(True)
        
        print(f"✅ Model initialized! Size: {model.get_memory_footprint() / 1e9:.2f}GB")
        
    except Exception as e:
        error_msg = f"❌ Error loading model: {str(e)}"
        status_manager.update_status("❌ Model loading failed", 0, error_msg)
        print(error_msg)

def start_model_loading():
    """Start model loading in background"""
    if status_manager.get_status()['model_loaded']:
        return "Model already loaded!"
        
    # Start background thread
    thread = threading.Thread(target=initialize_model_background, daemon=True)
    thread.start()
    
    return "🚀 Started loading model in background..."

def prepare_model_for_training():
    """Prepare model for training"""
    global model
    
    state = status_manager.get_status()
    if not state['model_loaded']:
        return "❌ Please load the model first!"
        
    if model is None:
        return "❌ Model not available!"
    
    try:
        status_manager.update_status("🔄 Configuring LoRA adapters...", 0)
        
        # Check if already prepared
        if hasattr(model, 'peft_config'):
            status_manager.update_status("✅ Model already prepared for training", 100)
            return "✅ Model already prepared for training"
        
        # Configure LoRA
        model = FastModel.get_peft_model(
            model,
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=32,
            lora_alpha=32,
            lora_dropout=0,
            bias="none",
            random_state=3407,
        )
        
        status_manager.update_status("✅ LoRA configuration applied!", 100)
        return "✅ LoRA configuration applied! Model ready for training."
        
    except Exception as e:
        error_msg = f"❌ Error preparing model: {str(e)}"
        status_manager.update_status("❌ Model preparation failed", 0, error_msg)
        return error_msg

def load_dataset_background():
    """Load dataset in background"""
    global train_dataset, eval_dataset
    
    try:
        status_manager.update_status("🔄 Loading Hausa medical dataset...", 10)
        
        dataset_name = "ictbiortc/hausa-medical-conversations-format-9k"
        dataset = load_dataset(dataset_name)
        
        if dataset is None or 'train' not in dataset or 'test' not in dataset:
            raise ValueError("Dataset not found or missing train/test splits")
        
        status_manager.update_status("🔄 Standardizing data formats...", 40)
        
        train_dataset = standardize_data_formats(dataset['train'])
        eval_dataset = standardize_data_formats(dataset['test'])
        
        if train_dataset is None or eval_dataset is None:
            raise ValueError("Failed to standardize dataset formats")
        
        status_manager.update_status("🔄 Applying chat template...", 70)
        
        def formatting_prompts_func(examples):
            convos = examples["conversations"]
            if convos is None:
                return {"text": []}
            texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False).removeprefix('<bos>') for convo in convos]
            return {"text": texts}
        
        train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
        eval_dataset = eval_dataset.map(formatting_prompts_func, batched=True)
        
        # Verify datasets were created successfully
        if train_dataset is None or eval_dataset is None:
            raise ValueError("Failed to apply chat template to datasets")
        
        status_manager.update_status(f"✅ Dataset loaded! Train: {len(train_dataset)}, Val: {len(eval_dataset)}", 100)
        
    except Exception as e:
        error_msg = f"❌ Error loading dataset: {str(e)}"
        status_manager.update_status("❌ Dataset loading failed", 0, error_msg)
        # Ensure global variables are set to None on failure
        train_dataset = None
        eval_dataset = None

def train_model_background(batch_size, grad_accum, epochs, lr):
    """Train model in background"""
    global model, tokenizer, trainer, training_stats, train_dataset, eval_dataset
    
    try:
        # Load dataset if needed
        if train_dataset is None or eval_dataset is None:
            status_manager.update_status("🔄 Loading dataset...", 5)
            load_dataset_background()
            
            # Check if dataset loading was successful
            if train_dataset is None or eval_dataset is None:
                error_msg = "❌ Failed to load dataset - dataset is None"
                status_manager.update_status("❌ Dataset loading failed", 0, error_msg)
                return
        
        status_manager.update_status("🔄 Setting up trainer...", 10)
        
        # Verify datasets have content
        if len(train_dataset) == 0 or len(eval_dataset) == 0:
            error_msg = "❌ Dataset is empty"
            status_manager.update_status("❌ Dataset is empty", 0, error_msg)
            return
        
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=SFTConfig(
                dataset_text_field="text",
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=grad_accum,
                warmup_steps=5,
                num_train_epochs=epochs,
                learning_rate=lr,
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                report_to="none",
                dataset_num_proc=2,
            ),
        )
        
        status_manager.update_status("🔄 Configuring response-only training...", 30)
        
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<start_of_turn>user\n",
            response_part="<start_of_turn>model\n",
        )
        
        status_manager.update_status("🚀 Starting training...", 50)
        
        # Start training
        trainer_stats = trainer.train()
        
        # Store results
        training_stats = trainer_stats
        status_manager.set_model_trained(True)
        
        # Set to inference mode
        FastModel.for_inference(model)
        
        training_time = trainer_stats.metrics['train_runtime']
        minutes = int(training_time // 60)
        seconds = int(training_time % 60)
        
        success_msg = f"🎉 Training completed! Loss: {trainer_stats.training_loss:.4f}, Time: {minutes}m {seconds}s"
        status_manager.update_status(success_msg, 100)
        
    except Exception as e:
        error_msg = f"❌ Training failed: {str(e)}"
        status_manager.update_status("❌ Training failed", 0, error_msg)

def start_training(batch_size, grad_accum, epochs, lr):
    """Start training in background"""
    state = status_manager.get_status()
    
    if not state['model_loaded']:
        return "❌ Please load the model first!"
        
    if model is None:
        return "❌ Model not available!"
        
    # Start training thread
    thread = threading.Thread(
        target=train_model_background,
        args=(batch_size, grad_accum, epochs, lr),
        daemon=True
    )
    thread.start()
    
    return "🚀 Started training in background..."

def get_current_status():
    """Get current status for polling"""
    state = status_manager.get_status()
    
    status_text = f"""📊 **Current Status**: {state['status']}
📈 **Progress**: {state['progress']}%
🤖 **Model Loaded**: {'✅' if state['model_loaded'] else '❌'}
🎓 **Model Trained**: {'✅' if state['model_trained'] else '❌'}"""

    if state['error']:
        status_text += f"\n❌ **Error**: {state['error']}"
        
    return status_text

def chat_with_model(message, history, temperature=1.0, max_tokens=200):
    """Chat with the model"""
    global model, tokenizer
    
    state = status_manager.get_status()
    if not state['model_loaded'] or model is None:
        return history + [{"role": "user", "content": message}, 
                         {"role": "assistant", "content": "❌ Please load the model first!"}]
    
    # System message
    system_message = "Kai likita ne mai hankali da ƙwarewa a fannin kiwon lafiya. Ka ba da shawarwari masu tushe a kimiyya, masu dacewa da al'adun Nijeriya."
    
    # Build messages
    messages = [{"role": "system", "content": system_message}]
    for msg in history:
        messages.append(msg)
    messages.append({"role": "user", "content": message})
    
    try:
        # Ensure inference mode
        if hasattr(model, 'peft_config'):
            FastModel.for_inference(model)
        
        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        
        # Generate
        inputs = tokenizer([text], return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.95,
                top_k=64,
                do_sample=True,
                repetition_penalty=1.1,
                use_cache=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # Decode response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        assistant_response = full_response.split("<start_of_turn>model\n")[-1].strip()
        
        return history + [{"role": "user", "content": message}, 
                         {"role": "assistant", "content": assistant_response}]
    
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        return history + [{"role": "user", "content": message}, 
                         {"role": "assistant", "content": error_msg}]

def save_model(output_path):
    """Save the trained model"""
    global model, tokenizer
    
    state = status_manager.get_status()
    if not state['model_trained']:
        return "❌ Please complete training first!"
        
    try:
        model.save_pretrained_merged(output_path, tokenizer)
        return f"✅ Model saved to {output_path}!"
    except Exception as e:
        return f"❌ Error saving: {str(e)}"

# Sample queries
SAMPLE_QUERIES = [
    "Ina jin ciwon kai da zazzabi tun kwana biyu. Me ya kamata in yi?",
    "Dana yana da gudawa sosai. Ina bukatan taimako.",
    "Yaya ake hana malaria lokacin damina?",
    "Ina da ciwon sukari. Wanne abinci ya dace da ni?",
]

# Create Gradio interface
with gr.Blocks(title="Hausa Health Assistant", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🏥 Hausa Health Assistant")
    gr.Markdown("Train and test an AI health assistant in Hausa language")
    
    # Status display (updates automatically)
    status_display = gr.Markdown("📊 **Status**: Ready")
    
    with gr.Tabs():
        # Model Management Tab
        with gr.TabItem("🤖 Model Management"):
            with gr.Row():
                with gr.Column():
                    load_btn = gr.Button("🚀 Load Base Model", variant="primary")
                    prep_btn = gr.Button("⚙️ Prepare for Training")
                    
                    gr.Markdown("### Training Parameters")
                    batch_size = gr.Slider(1, 8, 2, step=1, label="Batch Size")
                    grad_accum = gr.Slider(1, 8, 4, step=1, label="Gradient Accumulation")
                    epochs = gr.Slider(1, 3, 1, step=1, label="Epochs")
                    learning_rate = gr.Slider(1e-5, 5e-4, 2e-4, label="Learning Rate")
                    
                    train_btn = gr.Button("🎯 Start Training", variant="primary")
                    
                    gr.Markdown("### Save Model")
                    save_path = gr.Textbox(value="gemma3-hausa-medical", label="Save Path")
                    save_btn = gr.Button("💾 Save Model")
                    
                with gr.Column():
                    operation_status = gr.Textbox(label="Operation Status", lines=3)
        
        # Chat Tab
        with gr.TabItem("💬 Chat"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Sample Queries")
                    for i, query in enumerate(SAMPLE_QUERIES):
                        btn = gr.Button(f"{query[:40]}...", size="sm")
                        
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(label="Health Assistant", height=400, type="messages")
                    msg = gr.Textbox(label="Your Message (Hausa)", placeholder="Ka rubuta tambayarka...")
                    
                    with gr.Row():
                        temp = gr.Slider(0.1, 2.0, 1.0, label="Temperature")
                        tokens = gr.Slider(50, 500, 200, label="Max Tokens")
                    
                    with gr.Row():
                        send_btn = gr.Button("📤 Send", variant="primary")
                        clear_btn = gr.Button("🗑️ Clear")

    # Event handlers
    load_btn.click(start_model_loading, outputs=[operation_status])
    prep_btn.click(prepare_model_for_training, outputs=[operation_status])
    train_btn.click(
        start_training,
        inputs=[batch_size, grad_accum, epochs, learning_rate],
        outputs=[operation_status]
    )
    save_btn.click(save_model, inputs=[save_path], outputs=[operation_status])
    
    send_btn.click(
        chat_with_model,
        inputs=[msg, chatbot, temp, tokens],
        outputs=[chatbot]
    ).then(lambda: "", outputs=[msg])
    
    clear_btn.click(lambda: [], outputs=[chatbot])
    
    # Auto-update status every 2 seconds
    status_timer = gr.Timer(value=2)
    status_timer.tick(get_current_status, outputs=[status_display])
    
    # Sample button handlers
    def create_handler(query):
        return lambda: query
    
    for i, (btn, query) in enumerate(zip([btn for btn in app.children if isinstance(btn, gr.Button)], SAMPLE_QUERIES)):
        if i < len(SAMPLE_QUERIES):
            btn.click(create_handler(query), outputs=[msg])

if __name__ == "__main__":
    print("🚀 Starting Hausa Health Assistant...")
    app.launch(
        server_name="0.0.0.0",
        server_port=6006,
        share=True,
        debug=False
    )