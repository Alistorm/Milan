# Milan - Mistral AI x Alan Healthcare Hackathon

Milan is our project for the Mistral AI x Alan Healthcare Hackathon, held in Paris on October 12-13, 2024. We're leveraging the power of Mistral AI and other cutting-edge technologies to create a multimodal AI-powered healthcare companion that simplifies healthcare navigation, provides personalized insights, and promotes proactive health management.

**Hackathon Link:** https://alaninsurance.notion.site/Mistral-AI-x-Alan-Hackathon-Event-Details-Hackers-11b1426e8be78025ac6fdee99f0dcac1


## Project Description

Navigating the healthcare system can be complex and overwhelming.  Patients struggle with information overload, difficulty scheduling appointments, and often lack personalized support. Milan addresses these challenges by offering a multimodal AI companion that understands and responds to user needs through voice, text, and even facial expressions.  By integrating with patient health records and wearables, Milan provides tailored insights and recommendations for condition management, preventive care, and treatment adherence.

Milan uses Mistral AI's large language model for natural language understanding and generation, enabling it to engage in natural conversations, answer health-related questions, and provide empathetic support.  Facial expression analysis using computer vision helps Milan understand the user's emotional state and tailor its responses accordingly.  Integration with health data platforms allows Milan to offer personalized insights and recommendations.


## Key Features

* Multimodal Interaction (voice, text, facial expressions)
* Personalized health insights and recommendations
* Integration with health records and wearables
* Proactive health monitoring and alerts
* Appointment scheduling and medication reminders
* Secure and private data handling


## Technical Implementation

* **Mistral AI:** We utilize the Mistral API for natural language understanding and generation, powering Milan's conversational abilities.  We are currently using the free tier and will contact organizers if we exceed the usage limits.  We are primarily using the `/generate` endpoint for text generation.
* **Hugging Face:** We use a Hugging Face Space to host and deploy our facial expression analysis model. [Link to Hugging Face Space - if applicable]
* **Google Cloud:** We leverage Google Cloud for data storage and backend infrastructure to ensure scalability and reliability.
* **Nebius:** We utilize the provided Nebius H100 GPU for computationally intensive tasks like model training and inference. We also use their object storage and network-attached storage.
* **Other Technologies:** Python, React, relevant data visualization libraries


## Setup and Installation

1. Clone the repository: `git clone [repository URL]`
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables:  Create a `.env` file and add your API keys and credentials for Mistral AI, Google Cloud, etc.
4. Run the application: `python app.py`


## Usage

1. Launch the application.
2. Interact with Milan using voice or text input.
3. Connect your health data sources (optional).
4. Receive personalized insights and recommendations.


## Challenges

Integrating various APIs and ensuring seamless communication between different components was challenging.  Data privacy and security were also a key concern, requiring careful implementation of data handling procedures.


## Future Work

* Improve the accuracy and personalization of the AI model.
* Expand integration with more health data sources.
* Develop a mobile application for increased accessibility.
* Implement advanced features like predictive health modeling.


## Contributing

We welcome contributions! Please fork the repository and submit pull requests.


## License

MIT License


## Acknowledgements

* Mistral AI
* Alan
* Hugging Face
* Google Cloud
* Nebius


## Devpost Submission

(https://devpost.com/software/milan-qhp3u4)
