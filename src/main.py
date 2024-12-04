from persona import Persona
from researcher import ProductResearcher
from config import PROJECT_ROOT
import os

def main():
    """
    Main function to demonstrate the product research tool.
    """
    # Create sample personas
    personas = [
        Persona(
            name="Sarah Chen",
            age=28,
            occupation="UX Designer",
            interests=["Design", "Technology", "Sustainability"],
            pain_points=["Limited time", "Need for efficient workflows"],
            tech_savviness=5,
            background="Works at a fast-paced startup, always looking for tools to improve productivity"
        ),
        Persona(
            name="Robert Miller",
            age=45,
            occupation="Small Business Owner",
            interests=["Business growth", "Family time", "Golf"],
            pain_points=["Budget conscious", "Not very tech-savvy"],
            tech_savviness=2,
            background="Runs a local retail store, struggles with digital transformation"
        )
    ]
    
    researcher = ProductResearcher()
    
    # Example product
    product_description = """
    ProductivityPro - An AI-powered task management app that automatically 
    prioritizes your tasks, schedules your day, and provides intelligent 
    reminders based on your working patterns. Features include voice commands, 
    integration with popular tools, and real-time collaboration capabilities.
    """
    
    audience_description = """
    Primary Audience:
Knowledge Workers & Professionals
Busy professionals who manage multiple tasks and projects
People who work in dynamic environments with changing priorities
Remote workers and digital nomads who need flexible scheduling
Tech-Savvy Users
Comfortable with AI-powered tools
Early adopters who appreciate automation
Users familiar with digital productivity tools
Team Leaders & Collaborators
People who work in collaborative environments
Project managers and team coordinators
Those who need to sync their schedules with others
Secondary Audience:
Small Business Owners
Entrepreneurs juggling multiple responsibilities
Business owners looking to optimize their time management
Students & Academics
    Graduate students managing research and deadlines
    Academic professionals balancing teaching and research
    """
    
    # Use absolute path for images
    image_path = os.path.join(PROJECT_ROOT, "assets", "screenshots", "product_screenshot.jpg")
    
    results = researcher.analyze_product(
        personas=personas,
        product_description=product_description,
        product_images=[image_path] if os.path.exists(image_path) else None
    )
    
    for persona_name, feedback in results.items():
        print(f"\n=== Feedback from {persona_name} ===")
        print(feedback)

if __name__ == "__main__":
    main() 