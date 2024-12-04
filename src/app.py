import streamlit as st
from persona import Persona
from researcher import ProductResearcher
from persona_generator import PersonaGenerator
from feedback_analyzer import FeedbackAnalyzer
from config import PROJECT_ROOT, STREAMLIT_STYLE
import os
from typing import List, Dict, Any
import json
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
import networkx as nx
from functools import lru_cache
import time

@lru_cache(maxsize=10)
def get_dimensions_for_audience(audience_description: str, provider: str) -> List[Dict]:
    """Cache and return dimensions for a given audience description."""
    generator = PersonaGenerator(provider=provider)
    return generator.identify_audience_dimensions(audience_description)["dimensions"]

def display_analysis_results(report: Dict[str, Any]):
    """Display analysis results in the Streamlit UI."""
    # Overall Sentiment
    sentiment = report["aggregated_insights"]["overall_sentiment"]
    col1, col2 = st.columns([1, 2])
    with col1:
        fig = go.Figure(data=[
            go.Indicator(
                mode="gauge+number",
                value=(sentiment["score"] + 1) * 50,  # Convert -1:1 to 0:100
                title={"text": "Overall Sentiment"},
                gauge={"axis": {"range": [0, 100]},
                      "bar": {"color": "darkblue"},
                      "steps": [
                          {"range": [0, 33], "color": "lightcoral"},
                          {"range": [33, 66], "color": "khaki"},
                          {"range": [66, 100], "color": "lightgreen"}
                      ]}
            )
        ])
        st.plotly_chart(fig)
    
    with col2:
        st.write("### Sentiment Summary")
        st.write(sentiment["summary"])
    
    # Key Strengths and Concerns
    col1, col2 = st.columns(2)
    with col1:
        st.write("### Key Strengths")
        for strength in report["aggregated_insights"]["key_strengths"]:
            st.markdown(f"✅ {strength}")
    
    with col2:
        st.write("### Common Concerns")
        for concern in report["aggregated_insights"]["common_concerns"]:
            st.markdown(f"⚠️ {concern}")
    
    # Market Segments
    st.write("### Market Segment Analysis")
    segments_df = pd.DataFrame(report["aggregated_insights"]["market_segments"])
    st.dataframe(segments_df)
    
    # Recommendations
    st.write("### Recommendations")
    recommendations = report["aggregated_insights"]["recommendations"]
    
    # Group recommendations by priority
    priority_colors = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
    
    for priority in ["High", "Medium", "Low"]:
        priority_recs = [r for r in recommendations if r["priority"] == priority]
        if priority_recs:
            st.write(f"#### {priority_colors[priority]} {priority} Priority")
            for rec in priority_recs:
                with st.expander(f"**{rec['area']}**"):
                    st.write(rec["suggestion"])

def display_dimension_analysis(dimensions: List[Dict]):
    """Display analysis of audience dimensions."""
    st.subheader("🎯 Audience Dimension Analysis")
    
    # Create importance chart
    importance_data = {
        "Dimension": [d["name"] for d in dimensions],
        "Importance": [d["importance"] for d in dimensions],
        "Values": [", ".join(d["values"]) for d in dimensions]
    }
    
    fig = px.bar(importance_data, 
                 x="Dimension", 
                 y="Importance",
                 color="Importance",
                 color_continuous_scale="Viridis",
                 title="Dimension Importance Analysis")
    st.plotly_chart(fig)
    
    # Display dimension details
    cols = st.columns(len(dimensions))
    for i, dim in enumerate(dimensions):
        with cols[i]:
            with st.container(border=True):
                st.markdown(f"### {dim['name']}")
                st.markdown(f"**Importance:** {'🔵' * dim['importance']}")
                st.markdown("**Description:**")
                st.write(dim["description"])
                st.markdown("**Values:**")
                for value in dim["values"]:
                    st.markdown(f"- {value}")

def display_persona_coverage(personas: List[Persona], dimensions: List[Dict]):
    """Display how personas cover different dimension values."""
    st.subheader("👥 Persona Coverage Analysis")
    
    # Create coverage matrix
    coverage_data = []
    for persona in personas:
        row = {"Persona": persona.name}
        # Extract dimension values from persona background and characteristics
        for dim in dimensions:
            # This is a simplified example - you'd need to implement actual value extraction
            value = "TBD"  # You'll need to implement this
            row[dim["name"]] = value
        coverage_data.append(row)
    
    # Display coverage table
    st.dataframe(pd.DataFrame(coverage_data))
    
    # Create dimension network graph
    G = nx.Graph()
    for dim in dimensions:
        G.add_node(dim["name"], size=dim["importance"] * 10)
        for value in dim["values"]:
            G.add_node(value, size=5)
            G.add_edge(dim["name"], value)
    
    # Use networkx spring layout
    pos = nx.spring_layout(G)
    
    # Create plotly figure using go.Figure
    fig = go.Figure()
    
    # Add edges
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += (x0, x1, None)
        edge_trace['y'] += (y0, y1, None)
    
    fig.add_trace(edge_trace)
    
    # Add nodes
    for node in G.nodes():
        node_trace = go.Scatter(
            x=[pos[node][0]],
            y=[pos[node][1]],
            mode='markers+text',
            hoverinfo='text',
            text=[node],
            textposition="top center",
            marker=dict(
                size=G.nodes[node].get('size', 5),
                line_width=2
            ),
            name=node
        )
        fig.add_trace(node_trace)
    
    fig.update_layout(
        title="Dimension Network Analysis",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        plot_bgcolor='white'
    )
    
    st.plotly_chart(fig)

def persona_management_page():
    """Page for managing and generating personas."""
    st.title("Persona Management")
    
    if 'personas' not in st.session_state:
        st.session_state.personas = []
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Define Target Audience")
        
        # Update provider selection
        provider = st.selectbox(
            "LLM Provider",
            ["Ollama (Local)", "Groq (Faster)", "OpenAI (More Accurate)"],
            help="Select the AI provider for persona generation"
        )
        provider = (
            "ollama" if "Ollama" in provider
            else "groq" if "Groq" in provider
            else "openai"
        )
        
        audience_description = st.text_area(
            "Describe Your Target Audience",
            value=st.session_state.get('audience_description', ''),
            height=150,
            placeholder="""Describe your target audience..."""
        )
        
        # Add analyze button
        if st.button("Analyze Audience", type="primary"):
            if audience_description:
                st.session_state.audience_description = audience_description
                
                # Get dimensions with loading indicator
                with st.spinner("Analyzing audience dimensions..."):
                    try:
                        dimensions = get_dimensions_for_audience(audience_description, provider)
                        st.session_state.current_dimensions = dimensions
                        display_dimension_analysis(dimensions)
                        st.session_state.last_dimensions_hash = hash(str(dimensions))
                        st.success("Analysis complete!")
                        
                        # Show generation options after analysis
                        tab1, tab2 = st.tabs(["Basic Generation", "Swarm Generation"])
                        
                        with tab1:
                            subcol1, subcol2 = st.columns(2)
                            with subcol1:
                                num_personas = st.number_input("Number of Personas", min_value=1, max_value=5, value=3)
                            with subcol2:
                                ensure_diversity = st.checkbox("Ensure Dimension Coverage", value=True)
                            
                            if st.button("Generate Personas", type="primary", use_container_width=True):
                                with st.spinner("Generating personas..."):
                                    try:
                                        generator = PersonaGenerator(provider=provider)
                                        new_personas = generator.generate_diverse_personas(
                                            count=num_personas,
                                            audience_description=audience_description,
                                            ensure_diversity=ensure_diversity
                                        )
                                        st.session_state.personas.extend(new_personas)
                                        st.success(f"Successfully generated {len(new_personas)} personas!")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Failed to generate personas: {str(e)}")
                        
                        with tab2:
                            create_swarm_generation_ui(dimensions, provider)
                            
                    except Exception as e:
                        st.error(f"Failed to analyze audience dimensions: {str(e)}")
            else:
                st.warning("Please provide an audience description first.")
        
        # Show existing analysis if available
        elif 'current_dimensions' in st.session_state:
            dimensions = st.session_state.current_dimensions
            display_dimension_analysis(dimensions)
            
            # Show generation options
            tab1, tab2 = st.tabs(["Basic Generation", "Swarm Generation"])
            
            with tab1:
                subcol1, subcol2 = st.columns(2)
                with subcol1:
                    num_personas = st.number_input("Number of Personas", min_value=1, max_value=5, value=3)
                with subcol2:
                    ensure_diversity = st.checkbox("Ensure Dimension Coverage", value=True)
                
                if st.button("Generate Personas", type="primary", use_container_width=True):
                    with st.spinner("Generating personas..."):
                        try:
                            generator = PersonaGenerator(provider=provider)
                            new_personas = generator.generate_diverse_personas(
                                count=num_personas,
                                audience_description=audience_description,
                                ensure_diversity=ensure_diversity
                            )
                            st.session_state.personas.extend(new_personas)
                            st.success(f"Successfully generated {len(new_personas)} personas!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to generate personas: {str(e)}")
            
            with tab2:
                create_swarm_generation_ui(dimensions, provider)
    
    with col2:
        st.header("Manual Input")
        if st.button("Add Manual Persona", use_container_width=True):
            st.session_state.show_manual_form = True
        
        if st.session_state.get('show_manual_form', False):
            with st.form("manual_persona"):
                name = st.text_input("Name")
                age = st.number_input("Age", min_value=18, max_value=100, value=30)
                occupation = st.text_input("Occupation")
                
                interests = st.text_area("Interests (one per line)")
                interests_list = [i.strip() for i in interests.split('\n') if i.strip()]
                
                pain_points = st.text_area("Pain Points (one per line)")
                pain_points_list = [p.strip() for p in pain_points.split('\n') if p.strip()]
                
                tech_savviness = st.slider("Technical Proficiency", 1, 5, 3)
                background = st.text_area("Background Story")
                
                col1, col2 = st.columns(2)
                with col1:
                    submit = st.form_submit_button("Add")
                with col2:
                    if st.form_submit_button("Cancel"):
                        st.session_state.show_manual_form = False
                        st.rerun()
                
                if submit:
                    if name and occupation and background:
                        new_persona = Persona(
                            name=name,
                            age=age,
                            occupation=occupation,
                            interests=interests_list,
                            pain_points=pain_points_list,
                            tech_savviness=tech_savviness,
                            background=background
                        )
                        st.session_state.personas.append(new_persona)
                        st.session_state.show_manual_form = False
                        st.success(f"Added persona: {name}")
                        st.rerun()
                    else:
                        st.error("Please fill in all required fields")
    
    # Display Current Personas
    st.header("Current Personas")
    if not st.session_state.personas:
        st.info("No personas created yet. Generate some personas or add them manually.")
    else:
        cols = st.columns(3)
        for i, persona in enumerate(st.session_state.personas):
            with cols[i % 3]:
                with st.container(border=True):
                    st.subheader(f"👤 {persona.name}")
                    st.write(f"**Age:** {persona.age}")
                    st.write(f"**Occupation:** {persona.occupation}")
                    st.write("**Interests:**")
                    for interest in persona.interests:
                        st.markdown(f"- {interest}")
                    st.write("**Pain Points:**")
                    for pain in persona.pain_points:
                        st.markdown(f"- {pain}")
                    st.write(f"**Tech Savviness:** {'⭐' * persona.tech_savviness}")
                    with st.expander("Background"):
                        st.write(persona.background)
                    if st.button("Delete", key=f"del_{i}"):
                        st.session_state.personas.pop(i)
                        st.rerun()
    
    # Navigation
    if st.session_state.personas:
        if st.button("Proceed to Analysis →", type="primary", use_container_width=True):
            st.session_state.page = "analysis"
            st.rerun()

def analysis_page():
    """Page for product analysis and results."""
    st.title("Product Analysis")
    
    # Back button
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("← Back to Personas"):
            st.session_state.page = "personas"
            st.rerun()
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Product Information")
        
        # Use the saved product description
        product_description = st.text_area(
            "Product Description",
            value=st.session_state.get('product_description', ''),
            height=200,
            placeholder="Describe your product here..."
        )
        
        uploaded_files = st.file_uploader(
            "Upload Product Images",
            accept_multiple_files=True,
            type=['png', 'jpg', 'jpeg']
        )
        
        image_paths = []
        if uploaded_files:
            for uploaded_file in uploaded_files:
                save_path = os.path.join(PROJECT_ROOT, "assets", "screenshots", uploaded_file.name)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                image_paths.append(save_path)
                st.image(uploaded_file, caption=uploaded_file.name, width=200)
        
        if st.button("Analyze Product", 
                    disabled=not product_description or not st.session_state.personas,
                    type="primary",
                    use_container_width=True):
            with st.spinner("Analyzing product with different personas..."):
                researcher = ProductResearcher()
                results = researcher.analyze_product(
                    personas=st.session_state.personas,
                    product_description=product_description,
                    product_images=image_paths if image_paths else None
                )
                st.session_state.results = results
                st.session_state.product_description = product_description
                st.success("Analysis complete!")
                st.rerun()
    
    with col2:
        st.header("Results")
        
        if not st.session_state.get('results'):
            st.info("Enter product information and click 'Analyze Product' to see results.")
            return
        
        tab1, tab2 = st.tabs(["Raw Feedback", "Analysis"])
        
        with tab1:
            for persona_name, feedback in st.session_state.results.items():
                with st.expander(f"💭 Feedback from {persona_name}", expanded=True):
                    st.write(feedback)
        
        with tab2:
            analyzer = FeedbackAnalyzer()
            report = analyzer.generate_report(
                product_description=st.session_state.product_description,
                feedback_data=st.session_state.results
            )
            display_analysis_results(report)
            
            if st.button("Export Report", type="primary"):
                export_data = {
                    "product_description": st.session_state.product_description,
                    "feedback": st.session_state.results,
                    "analysis": report
                }
                st.download_button(
                    "Download Full Report (JSON)",
                    data=json.dumps(export_data, indent=2),
                    file_name="user_research_report.json",
                    mime="application/json"
                )

def estimate_generation_time(count: int, batch_size: int, provider: str) -> str:
    """
    Estimate the time needed for persona generation.
    """
    # Average time per persona (in seconds) based on provider
    time_per_persona = {
        "groq": 3,     # Groq is faster
        "openai": 5,   # OpenAI is slower
        "ollama": 2    # Local inference can be faster
    }
    
    # Get the time estimate for the provider
    per_persona_time = time_per_persona.get(provider, 4)  # Default to 4 seconds if unknown
    
    # Calculate base time
    base_time = count * per_persona_time
    
    # Add overhead for batching
    num_batches = (count + batch_size - 1) // batch_size
    batch_overhead = num_batches * 1  # 1 second overhead per batch
    
    # Add buffer for variations (±20%)
    min_time = int((base_time + batch_overhead) * 0.8)
    max_time = int((base_time + batch_overhead) * 1.2)
    
    # Format time range
    if min_time < 60:
        return f"{min_time}-{max_time} seconds"
    else:
        min_minutes = min_time // 60
        max_minutes = max_time // 60
        return f"{min_minutes}-{max_minutes} minutes"

def create_swarm_generation_ui(dimensions: List[Dict], provider: str):
    """Create UI for swarm generation with probability distributions."""
    st.subheader("🐝 Swarm Generation Settings")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        swarm_size = st.number_input(
            "Number of Personas to Generate",
            min_value=10,
            max_value=100,
            value=20,
            help="How many personas to generate in the swarm"
        )
        
        distribution_type = st.selectbox(
            "Distribution Type",
            ["Uniform", "Custom Weights"],
            help="How to distribute personas across dimension values"
        )
    
    with col2:
        batch_size = st.number_input(
            "Batch Size",
            min_value=1,
            max_value=20,
            value=1,
            help="How many personas to generate at once"
        )
        
        # Show time estimate
        estimated_time = estimate_generation_time(swarm_size, batch_size, provider)
        st.info(f"Estimated time: {estimated_time}")
    
    weights = {}
    if distribution_type == "Custom Weights":
        st.write("### Dimension Value Weights")
        st.info("Adjust the relative importance of each value. Values will be normalized during generation.")
        
        # Create tabs for dimensions
        tabs = st.tabs([dim["name"] for dim in dimensions])
        
        for i, (tab, dim) in enumerate(zip(tabs, dimensions)):
            with tab:
                # Initialize weights dict for this dimension
                dim_weights = {}
                
                # Create columns for the dimension values
                cols = st.columns(len(dim["values"]))
                
                # Create a slider for each value with unique keys
                for j, (col, value) in enumerate(zip(cols, dim["values"])):
                    with col:
                        st.write(f"**{value}**")
                        weight = st.slider(
                            "Relative Weight",
                            min_value=0,
                            max_value=100,
                            value=100 // len(dim["values"]),  # Equal default weights
                            label_visibility="collapsed",
                            key=f"weight_{i}_{j}_{dim['name']}_{value}"
                        )
                        dim_weights[value] = weight
                
                weights[dim["name"]] = dim_weights
    
    # Generate button
    if st.button("Generate Swarm", type="primary", use_container_width=True):
        progress_bar = st.progress(0, "Preparing to generate personas...")
        status_text = st.empty()
        time_text = st.empty()
        
        try:
            generator = PersonaGenerator(provider=provider)
            personas = []
            
            # Create normalized settings
            normalized_settings = {}
            if distribution_type == "Custom Weights":
                for dim_name, dim_weights in weights.items():
                    total = sum(dim_weights.values())
                    if total > 0:
                        for value, weight in dim_weights.items():
                            normalized_settings[f"{dim_name}_{value}"] = weight / total
            
            # Generate personas with time tracking
            start_time = time.time()
            
            for batch_start in range(0, swarm_size, batch_size):
                batch_end = min(batch_start + batch_size, swarm_size)
                
                # Update status with time information
                elapsed = time.time() - start_time
                personas_generated = len(personas)
                if personas_generated > 0:
                    avg_time_per_persona = elapsed / personas_generated
                    remaining_personas = swarm_size - personas_generated
                    estimated_remaining = remaining_personas * avg_time_per_persona
                    time_text.text(f"Time remaining: {int(estimated_remaining)} seconds")
                
                status_text.text(f"Generating personas {batch_start + 1} to {batch_end}...")
                
                batch_personas = generator.generate_diverse_personas(
                    count=batch_size,
                    audience_description=st.session_state.audience_description,
                    ensure_diversity=False,
                    distribution_settings=normalized_settings if distribution_type == "Custom Weights" else None
                )
                
                personas.extend(batch_personas)
                progress = (batch_end) / swarm_size
                progress_bar.progress(progress, f"Generated {batch_end} of {swarm_size} personas...")
            
            # Show final timing
            total_time = time.time() - start_time
            time_text.text(f"Total generation time: {int(total_time)} seconds")
            
            st.session_state.personas = personas
            status_text.text("Generation complete!")
            progress_bar.progress(1.0, "✅ Swarm generation complete!")
            
            # Show final distribution analysis
            if distribution_type == "Custom Weights":
                st.subheader("Generated Distribution Analysis")
                
                for dim_name, dim_weights in weights.items():
                    st.write(f"### {dim_name}")
                    
                    # Count actual distribution
                    value_counts = {value: 0 for value in dim_weights.keys()}
                    total_personas = len(personas)
                    
                    for persona in personas:
                        for line in persona.background.split('\n'):
                            if f"{dim_name}:" in line:
                                value = line.split(':')[1].strip()
                                if value in value_counts:
                                    value_counts[value] += 1
                    
                    # Create comparison data
                    comparison_data = []
                    for value in dim_weights.keys():
                        target = normalized_settings.get(f"{dim_name}_{value}", 1.0/len(dim_weights))
                        actual = value_counts[value]/total_personas if total_personas > 0 else 0
                        comparison_data.extend([
                            {"Value": value, "Distribution": target, "Type": "Target"},
                            {"Value": value, "Distribution": actual, "Type": "Actual"}
                        ])
                    
                    # Show comparison chart
                    fig = px.bar(pd.DataFrame(comparison_data),
                               x="Value",
                               y="Distribution",
                               color="Type",
                               barmode="group",
                               title="Target vs Actual Distribution")
                    fig.update_layout(yaxis_tickformat=".0%")
                    st.plotly_chart(fig)
            
            st.success(f"Successfully generated {len(personas)} personas!")
            st.rerun()
            
        except Exception as e:
            st.error(f"Failed to generate swarm: {str(e)}")

def main():
    """Main application."""
    st.set_page_config(page_title="LLM User Research Tool", layout="wide")
    st.markdown(STREAMLIT_STYLE, unsafe_allow_html=True)
    
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = "personas"
    
    # Display current page
    if st.session_state.page == "personas":
        persona_management_page()
    else:
        analysis_page()

if __name__ == "__main__":
    main() 