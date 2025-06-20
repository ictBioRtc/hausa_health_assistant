# Fine-tune Hausa Health Assistant AI Model

This workshop demonstrates how to fine-tune a multilingual AI model to create a health assistant that responds in Hausa language, providing culturally appropriate medical guidance for Nigerian communities.

## Prerequisites

### 1. GPU Requirements
- Recommended: GPU with at least 16GB VRAM
- Minimum: GPU with 8GB VRAM (will use smaller batch sizes)
- CPU training is possible but very slow

### 2. HuggingFace Setup (Optional - for saving models)
1. Create an account at [huggingface.co](https://huggingface.co)
2. Go to Settings → Access Tokens
3. Create a new token with write access (needed for uploading)
4. Copy and save your token - you'll need it later

## Quick Start

1. Open terminal:
   ```bash
   File > New Launcher > Terminal
   ```

2. Clone the repository:
   ```bash
   git clone https://github.com/ictBioRtc/hausa_health_assistant.git
   ```

3. Navigate to project directory:
   ```bash
   cd hausa-health-assistant
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Run the application:
   ```bash
   python app.py --share
   ```

6. Copy the public URL provided (e.g., https://abc123.gradio.live)
7. Open in a new browser tab

## Using the Application

### Step 1: Initialize the Base Model
1. Go to "🤖 Model Management" tab
2. Click "🚀 Load Base Model" and wait for completion
3. Once loaded, click "⚙️ Prepare for Training"

### Step 2: Test the Untrained Model (Optional)
1. Go to "💬 Chat" tab
2. Try sample queries or type your own in Hausa:
   - "Ina jin ciwon kai da zazzabi tun kwana biyu. Me ya kamata in yi?"
   - "Dana yana da gudawa sosai. Ina bukatan taimako."
3. Observe how the base model responds (before training)

### Step 3: Train the Model
1. Return to "🤖 Model Management" tab
2. Adjust training parameters:
   - **Epochs**: Start with 1 (full training run)
   - **Batch Size**: Use 2 (adjust based on GPU memory)
   - **Gradient Accumulation**: Keep default 4
   - **Learning Rate**: Keep default 2e-4
3. Click "🎯 Start Training"
4. Monitor the status display for progress updates
5. Training typically takes 15-30 minutes for 1 epoch

### Step 4: Test Your Trained Model
1. Go to "💬 Chat" tab after training completes
2. Test the same queries you used before
3. Compare the quality and cultural appropriateness of responses
4. Notice the improvement in medical knowledge and Hausa fluency

### Step 5: Save Your Model (Optional)
1. In "🤖 Model Management" tab, enter a save path
2. Click "💾 Save Model" to save locally

## Sample Hausa Medical Queries

The application includes sample query buttons that demonstrate the model's capabilities:

- **"Ina jin ciwon kai da zazzabi tun kwana biyu. Me ya kamata in yi?"**
  - *Translation: "I have headache and fever for two days. What should I do?"*

- **"Dana yana da gudawa sosai. Ina bukatan taimako."**
  - *Translation: "My child has severe diarrhea. I need help."*

- **"Yaya ake hana malaria lokacin damina?"**
  - *Translation: "How do you prevent malaria during rainy season?"*

- **"Ina da ciwon sukari. Wanne abinci ya dace da ni?"**
  - *Translation: "I have diabetes. What food is suitable for me?"*

## Interface Overview

### Model Management Tab (🤖)
- **Load Base Model**: Downloads and initializes Gemma-3 4B model
- **Prepare for Training**: Configures LoRA adapters for efficient fine-tuning
- **Training Parameters**: Adjustable batch size, epochs, learning rate, gradient accumulation
- **Start Training**: Begins fine-tuning on Hausa medical dataset
- **Save Model**: Exports trained model to specified path
- **Operation Status**: Shows detailed progress and error messages

### Chat Tab (💬)
- **Sample Queries**: Pre-written Hausa medical questions as buttons
- **Chat Interface**: Interactive conversation with the AI assistant
- **Temperature Control**: Adjusts response creativity (0.1-2.0)
- **Max Tokens**: Controls response length (50-500)
- **Real-time Status**: Shows model loading and training status

## About the Dataset

This project uses the `ictbiortc/hausa-medical-conversations-format-9k` dataset, which contains:
- 8,100 training conversations
- 900 validation conversations
- Medical Q&A pairs in Hausa language
- Culturally appropriate responses for Nigerian healthcare context

## Model Architecture

- **Base Model**: Gemma-3 4B Instruct (Google's multilingual model)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Training Strategy**: Response-only training (only learns from assistant responses)
- **Memory Efficiency**: 4-bit quantization available for reduced GPU usage
- **System Prompt**: Hidden system prompt ensures medical expert persona

## Key Features

### Background Processing
- Model loading and training happen in separate threads
- Interface remains responsive during long operations
- Real-time status updates every 2 seconds

### Memory Management
- Automatic CUDA cache clearing
- Optimized for GPU memory efficiency
- Thread-safe operations for concurrent access

### User Experience
- Clean, intuitive interface with tab-based navigation
- Sample queries for easy testing
- Hidden system prompt maintains professional medical context
- Seamless transition from untrained to trained model

## Applications & Impact

This Hausa health assistant can be used for:

### Healthcare Access
- **Rural Healthcare**: Provide medical guidance in remote areas with limited healthcare access
- **Telemedicine**: Support Hausa-speaking patients in digital health consultations
- **Health Education**: Disseminate medical information in native language
- **Emergency Guidance**: Provide first-aid instructions in critical situations

### Community Health
- **Maternal Health**: Support pregnant women with culturally appropriate advice
- **Child Healthcare**: Guide parents on pediatric health issues
- **Preventive Care**: Educate communities about disease prevention
- **Mental Health**: Provide culturally sensitive mental health support

### Integration Opportunities
- **Mobile Health Apps**: Integrate into existing mHealth platforms
- **WhatsApp Bots**: Deploy as conversational AI on popular messaging platforms
- **Healthcare Hotlines**: Support call centers with AI-assisted responses
- **Educational Tools**: Use in medical training and health literacy programs

## Training Details

### Model Configuration
- **LoRA Rank (r)**: 32 (higher rank for better medical knowledge retention)
- **LoRA Alpha**: 32 (recommended to match rank)
- **Target Modules**: All attention and MLP layers
- **Gradient Accumulation**: 4 steps (effective batch size of 8)
- **Optimizer**: AdamW 8-bit for memory efficiency

### Training Tips
- **For Quick Testing**: Use 1 epoch (15-30 minutes)
- **For Production Use**: Consider 2-3 epochs for better performance
- **Memory Issues**: Reduce batch size to 1 if out of memory
- **Better Results**: Increase LoRA rank to 64 for more complex medical reasoning

## Troubleshooting

### Common Issues

**1. "CUDA out of memory"**
- Reduce batch size to 1
- Restart the application to clear GPU cache
- Close other GPU-intensive applications

**2. "Model not loading"**
- Check internet connection for base model download (~8GB)
- Verify HuggingFace Hub access
- Ensure sufficient disk space
- Wait for background loading to complete

**3. "Dataset loading failed"**
- Verify internet connection
- Check if dataset is publicly accessible
- Check the Operation Status for detailed error messages

**4. "Training too slow"**
- Verify GPU is being used (check device output in logs)
- Reduce sequence length if needed
- Consider using smaller batch sizes

**5. "Poor response quality"**
- Train for more epochs (2-3 instead of 1)
- Increase LoRA rank for better learning capacity
- Verify training completed successfully
- Check that model switched to inference mode

### Performance Optimization

**GPU Memory Management:**
- The app automatically clears CUDA cache
- Monitor memory usage in system resources
- Use `nvidia-smi` in terminal to check GPU utilization

**Training Speed:**
- Gradient accumulation is used instead of larger batch sizes
- Mixed precision training handled automatically by unsloth
- Background processing keeps interface responsive

## Cultural Considerations

### Medical Ethics in Hausa Context
- Always recommends consulting qualified medical professionals
- Respects traditional healing practices while promoting modern medicine
- Provides culturally sensitive advice for sensitive topics
- Considers Islamic principles in healthcare guidance

### Language Authenticity
- Uses authentic Hausa medical terminology
- Incorporates appropriate cultural greetings and courtesies
- Adapts explanations to local understanding and context
- Avoids direct translations that may be culturally inappropriate

### System Prompt Design
The hidden system prompt instructs the model to be:
- A careful and skilled doctor in healthcare
- Provide science-based advice
- Ensure cultural appropriateness for Nigerian context

## Deployment Options

### Local Deployment
```bash
# Run with default settings
python app.py

# Run with sharing enabled
python app.py --share

# Custom port (if needed)
# Modify server_port in gradio_app.py
```

### Cloud Deployment
- **Hugging Face Spaces**: Easy deployment with Gradio interface
- **Google Colab**: Free GPU access for development and training
- **AWS/Azure**: Production deployment with autoscaling
- **Docker**: Containerized deployment for consistency

## Research & Development

### Future Improvements
- **Multimodal Capabilities**: Add image understanding for medical images
- **Voice Integration**: Support Hausa speech recognition and synthesis
- **Knowledge Graphs**: Integrate structured medical knowledge
- **Multi-language Support**: Extend to other Nigerian languages

### Academic Applications
- Research on low-resource language AI in healthcare
- Cultural adaptation of medical AI systems
- Evaluation frameworks for multilingual health AI
- Ethics of AI in traditional medicine contexts

## Contributing

### How to Contribute
1. Fork the repository
2. Add new training data or improve existing datasets
3. Enhance model architecture or training procedures
4. Improve user interface and experience
5. Submit pull requests with detailed documentation

### Code of Conduct
- Respect cultural sensitivities and medical ethics
- Ensure all contributions maintain patient privacy
- Follow responsible AI development practices
- Collaborate with healthcare professionals for validation

## License & Ethics

### Responsible Use
- This model is for educational and research purposes
- Always consult qualified healthcare professionals for medical decisions
- Do not use for emergency medical situations
- Respect patient privacy and data protection laws

### Limitations
- Not a replacement for professional medical advice
- May have biases from training data
- Limited to general health guidance
- Requires regular updates with latest medical knowledge

## Support & Resources

### Documentation
- [Unsloth Documentation](https://docs.unsloth.ai/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Gradio Documentation](https://gradio.app/docs/)

### Community
- Join our Discord for technical support
- Follow @ictBioRtc on Twitter for updates
- Contribute to our GitHub repository
- Share your results and improvements

### Citation
If you use this work in research, please cite:
```bibtex
@software{hausa_health_assistant_2024,
  title={Hausa Health Assistant: Fine-tuning Multilingual AI for Healthcare},
  author={ICT BioRTC},
  year={2024},
  url={https://github.com/ictBioRtc/hausa-health-assistant}
}
```

---

**⚠️ Important Disclaimer**: This AI assistant is designed for educational and informational purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified healthcare providers with any questions regarding medical conditions.
