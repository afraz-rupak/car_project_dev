import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import pandas as pd
import json
import os
import glob
from pathlib import Path

# Configure page
st.set_page_config(
    page_title="Car Classification System",
    page_icon="🚗",
    layout="wide"
)

@st.cache_data
def load_config():
    """Load model configuration"""
    config_path = Path("/Users/afrazrupak/CarProject_dev/model_config.json")
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}

@st.cache_data
def extract_model_labels_from_dataset():
    """Extract actual model labels from the dataset structure"""
    model_labels = {}
    base_path = Path("/Users/afrazrupak/CarProject_dev/data/yolo_sample/crop_images/train")
    
    if base_path.exists():
        for make_dir in base_path.iterdir():
            if make_dir.is_dir():
                make_id = make_dir.name
                model_labels[make_id] = []
                
                for model_dir in make_dir.iterdir():
                    if model_dir.is_dir():
                        model_id = model_dir.name
                        model_labels[make_id].append(model_id)
                
                # Sort model IDs for consistent indexing
                model_labels[make_id] = sorted(model_labels[make_id])
    
    return model_labels

@st.cache_data
def load_car_make_model_mappings():
    """Load car make and model name mappings from CSV files if available"""
    mappings = {
        'make_names': {},
        'model_names': {}
    }
    
    base_path = Path("/Users/afrazrupak/CarProject_dev")
    
    # Try to load make names
    make_csv_paths = [
        base_path / "data" / "CompCar" / "data" / "misc" / "make_names.csv",
        base_path / "data" / "car_csv" / "label_vocabulary_top20_make.csv"
    ]
    
    for csv_path in make_csv_paths:
        if csv_path.exists():
            try:
                df_make = pd.read_csv(csv_path)
                if 'make_id' in df_make.columns and 'make_names' in df_make.columns:
                    mappings['make_names'] = dict(zip(
                        df_make['make_id'].astype(str), 
                        df_make['make_names'].astype(str)
                    ))
                    break
            except Exception as e:
                st.sidebar.warning(f"Could not load make names from {csv_path}: {e}")
    
    # Try to load model names
    model_csv_paths = [
        base_path / "data" / "CompCar" / "data" / "misc" / "make_model_final.csv",
        base_path / "data" / "car_csv" / "model_names.csv"
    ]
    
    for csv_path in model_csv_paths:
        if csv_path.exists():
            try:
                df_model = pd.read_csv(csv_path)
                if 'model_id' in df_model.columns and 'model_names' in df_model.columns:
                    mappings['model_names'] = dict(zip(
                        df_model['model_id'].astype(str), 
                        df_model['model_names'].astype(str)
                    ))
                    break
            except Exception as e:
                st.sidebar.warning(f"Could not load model names from {csv_path}: {e}")
    
    return mappings

@st.cache_resource
def load_models():
    """Load YOLO models based on configuration"""
    config = load_config()
    base_path = Path("/Users/afrazrupak/CarProject_dev")
    
    models = {}
    
    try:
        # Load general car make classifier
        general_make_paths = [
            base_path / "runs" / "classify" / "yolo11n_cls_100e" / "weights" / "best.pt",
            base_path / "runs" / "classify2" / "yolo11n_cls_100e" / "weights" / "best.pt",
            base_path / "runs" / "classify3" / "yolo11n_cls_100e" / "weights" / "best.pt"
        ]
        
        for path in general_make_paths:
            if path.exists():
                models['car_make_general'] = YOLO(str(path))
                st.sidebar.success(f"✅ General car make model loaded from {path.name}")
                break
        
        # Load body type classifier
        body_type_path = base_path / "runs" / "car_type_classify" / "yolo11n_cls_100e" / "weights" / "best.pt"
        if body_type_path.exists():
            models['body_type'] = YOLO(str(body_type_path))
            st.sidebar.success("✅ Body type model loaded")
        
        # Load specialized car make models (lazy loading)
        models['specialized_models'] = {}
        available_makes = []
        
        # Scan for available specialized models
        runs_dir = base_path / "runs" / "Tee_classify_top15"
        if runs_dir.exists():
            for model_dir in runs_dir.iterdir():
                if model_dir.is_dir() and model_dir.name.startswith("model_cls_"):
                    # Extract make ID from folder name: model_cls_100_y11n_100e
                    parts = model_dir.name.split("_")
                    if len(parts) >= 3:
                        make_id = parts[2]
                        weights_path = model_dir / "weights" / "best.pt"
                        if weights_path.exists():
                            available_makes.append(make_id)
        
        st.sidebar.info(f"📋 {len(available_makes)} specialized models available")
        
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
    
    return models, config

@st.cache_data
def load_label_vocabularies():
    """Load label vocabularies for each classification task"""
    base_path = Path("/Users/afrazrupak/CarProject_dev")
    
    labels = {}
    
    # Load car make labels
    make_labels_path = base_path / "yolo_cls_car_makes_front_rear" / "label_vocabulary.csv"
    if make_labels_path.exists():
        make_df = pd.read_csv(make_labels_path)
        labels['car_makes'] = make_df['class'].tolist()
    
    # Load body type labels
    body_type_labels_path = base_path / "yolo_cls_car_type" / "label_vocabulary.csv"
    if body_type_labels_path.exists():
        body_df = pd.read_csv(body_type_labels_path)
        labels['body_types'] = body_df['class'].tolist()
    
    return labels

def load_specialized_model(make_id, models, config):
    """Load specialized model for specific car make"""
    if make_id not in models['specialized_models']:
        base_path = Path("/Users/afrazrupak/CarProject_dev")
        model_path = base_path / "runs" / "Tee_classify_top15" / f"model_cls_{make_id}_y11n_100e" / "weights" / "best.pt"
        
        if model_path.exists():
            models['specialized_models'][make_id] = YOLO(str(model_path))
            st.sidebar.success(f"✅ Loaded specialized model for make {make_id}")
        else:
            st.sidebar.warning(f"⚠️ Specialized model not found for make {make_id}")
    
    return models['specialized_models'].get(make_id)

def predict_car_make(image, model, labels):
    """Predict car make using general classifier"""
    try:
        prediction = model(image, verbose=False)
        if prediction and len(prediction) > 0:
            probs = prediction[0].probs
            if probs is not None:
                top_class_idx = probs.top1
                confidence = probs.top1conf.item()
                
                if top_class_idx < len(labels['car_makes']):
                    predicted_make = labels['car_makes'][top_class_idx]
                    return predicted_make, confidence, top_class_idx
    except Exception as e:
        st.error(f"Error predicting car make: {str(e)}")
    
    return None, 0.0, None

def predict_body_type(image, model, labels):
    """Predict car body type"""
    try:
        prediction = model(image, verbose=False)
        if prediction and len(prediction) > 0:
            probs = prediction[0].probs
            if probs is not None:
                top_class_idx = probs.top1
                confidence = probs.top1conf.item()
                
                if top_class_idx < len(labels['body_types']):
                    predicted_type = labels['body_types'][top_class_idx]
                    return predicted_type, confidence
    except Exception as e:
        st.error(f"Error predicting body type: {str(e)}")
    
    return None, 0.0

def predict_specialized_model(image, specialized_model, make_id, model_labels, name_mappings):
    """Use specialized model to get detailed car model predictions"""
    try:
        prediction = specialized_model(image, verbose=False)
        if prediction and len(prediction) > 0:
            probs = prediction[0].probs
            if probs is not None:
                # Get top 3 predictions
                top_indices = probs.top5[:3] if len(probs.top5) >= 3 else probs.top5
                top_confs = probs.top5conf[:3] if len(probs.top5conf) >= 3 else probs.top5conf
                
                results = []
                available_models = model_labels.get(str(make_id), [])
                
                for idx, conf in zip(top_indices, top_confs):
                    # Map class index to actual model ID
                    if idx < len(available_models):
                        model_id = available_models[idx]
                        # Try to get model name from mappings, fallback to ID
                        model_name = name_mappings['model_names'].get(model_id, f"Model {model_id}")
                        
                        results.append({
                            'model_id': model_id,
                            'model_name': model_name,
                            'confidence': conf.item()
                        })
                    else:
                        results.append({
                            'model_id': f'Unknown_{idx}',
                            'model_name': f'Unknown Model {idx}',
                            'confidence': conf.item()
                        })
                
                return results
    except Exception as e:
        st.error(f"Error with specialized model: {str(e)}")
    
    return []

# Main Streamlit App
def main():
    st.title("🚗 Advanced Car Classification System")
    st.markdown("---")
    
    st.markdown("""
    ### Multi-Stage Car Classification System
    This system provides comprehensive car identification:
    1. **Car Make Detection**: Identifies the manufacturer
    2. **Specific Model Analysis**: Uses brand-specific models to identify exact car models
    3. **Body Type Classification**: Identifies vehicle body style
    
    **Features:**
    - 🏭 **Car Make**: 14+ supported brands
    - 🚗 **Car Models**: Specific model identification per brand
    - 🔧 **Body Type**: 11 different vehicle types
    """)
    
    # Load models and data
    with st.spinner("🔄 Loading AI models and mappings..."):
        models, config = load_models()
        labels = load_label_vocabularies()
        model_labels = extract_model_labels_from_dataset()
        name_mappings = load_car_make_model_mappings()
    
    if not models:
        st.error("❌ No models could be loaded. Please check model paths.")
        return
    
    # Sidebar with system information
    with st.sidebar:
        st.header("🛠️ System Status")
        
        if 'car_make_general' in models:
            st.success("✅ General Make Classifier")
        if 'body_type' in models:
            st.success("✅ Body Type Classifier")
        
        st.header("📊 Available Data")
        st.info(f"🏭 Car Makes: {len(labels.get('car_makes', []))}")
        st.info(f"🔧 Body Types: {len(labels.get('body_types', []))}")
        
        total_models = sum(len(models) for models in model_labels.values())
        st.info(f"🚗 Total Models: {total_models}")
        
        if name_mappings['make_names']:
            st.success(f"✅ Make names loaded: {len(name_mappings['make_names'])}")
        if name_mappings['model_names']:
            st.success(f"✅ Model names loaded: {len(name_mappings['model_names'])}")
    
    # Main content
    st.markdown("---")
    uploaded_file = st.file_uploader(
        "📸 Upload a car image for comprehensive analysis", 
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload a clear image of a car for best results. Front/rear views work best for make identification."
    )
    
    if uploaded_file is not None:
        # Create layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Display image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Image info
            with st.expander("📋 Image Details"):
                st.write(f"**Size:** {image.size[0]} × {image.size[1]} pixels")
                st.write(f"**Mode:** {image.mode}")
                st.write(f"**Format:** {uploaded_file.type}")
                st.write(f"**File size:** {len(uploaded_file.getvalue()) / 1024:.1f} KB")
        
        with col2:
            st.markdown("### 🎯 Analysis Results")
            
            # Convert image for processing
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Initialize results
            predicted_make = None
            make_id = None
            
            # Step 1: General car make prediction
            with st.spinner("🔍 Step 1: Identifying car make..."):
                if 'car_make_general' in models:
                    predicted_make, make_confidence, make_class_idx = predict_car_make(
                        image, models['car_make_general'], labels
                    )
                    
                    if predicted_make:
                        # Map make name to ID for specialized model lookup
                        make_name_to_id = {v.lower(): k for k, v in name_mappings['make_names'].items()}
                        make_id = make_name_to_id.get(predicted_make.lower())
                        
                        # Display make with proper name
                        display_make = name_mappings['make_names'].get(str(make_class_idx), predicted_make)
                        
                        st.success(f"🏭 **Car Make:** {display_make}")
                        st.progress(make_confidence)
                        st.caption(f"Confidence: {make_confidence:.1%}")
                        
                        # Step 2: Use specialized model for specific model identification
                        if make_id:
                            with st.spinner(f"🔬 Step 2: Identifying specific {display_make} model..."):
                                specialized_model = load_specialized_model(make_id, models, config)
                                
                                if specialized_model:
                                    model_results = predict_specialized_model(
                                        image, specialized_model, make_id, model_labels, name_mappings
                                    )
                                    
                                    if model_results:
                                        st.markdown("#### 🎯 Specific Car Model")
                                        for i, result in enumerate(model_results[:3]):
                                            rank = ["🥇", "🥈", "🥉"][i]
                                            confidence = result['confidence']
                                            model_name = result['model_name']
                                            model_id_display = result['model_id']
                                            
                                            # Color code confidence
                                            if confidence > 0.8:
                                                conf_color = "🟢"
                                            elif confidence > 0.6:
                                                conf_color = "🟡"
                                            else:
                                                conf_color = "🔴"
                                            
                                            st.write(f"{rank} **{model_name}** (ID: {model_id_display})")
                                            st.progress(confidence)
                                            st.caption(f"{conf_color} Confidence: {confidence:.1%}")
                                            
                                            if i == 0:  # Store top prediction for summary
                                                top_model = model_name
                                                top_model_id = model_id_display
                                else:
                                    st.info("🔄 Specialized model available but no predictions generated")
                        else:
                            st.warning("⚠️ No specialized model available for this make")
                    else:
                        st.warning("⚠️ Could not identify car make")
            
            # Step 3: Body type prediction
            with st.spinner("🔍 Step 3: Analyzing body type..."):
                if 'body_type' in models:
                    predicted_body_type, body_confidence = predict_body_type(
                        image, models['body_type'], labels
                    )
                    
                    if predicted_body_type:
                        st.success(f"🔧 **Body Type:** {predicted_body_type.title()}")
                        st.progress(body_confidence)
                        st.caption(f"Confidence: {body_confidence:.1%}")
                    else:
                        st.warning("⚠️ Could not identify body type")
            
            # Summary section
            st.markdown("---")
            st.markdown("### 📊 Complete Vehicle Profile")
            
            summary_data = []
            if predicted_make:
                display_make = name_mappings['make_names'].get(str(make_class_idx), predicted_make)
                summary_data.append(["Car Make", display_make, f"{make_confidence:.1%}"])
            
            if 'top_model' in locals():
                summary_data.append(["Car Model", f"{top_model} (ID: {top_model_id})", f"{model_results[0]['confidence']:.1%}"])
            
            if 'predicted_body_type' in locals() and predicted_body_type:
                summary_data.append(["Body Type", predicted_body_type.title(), f"{body_confidence:.1%}"])
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data, columns=["Attribute", "Prediction", "Confidence"])
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
                
                # Vehicle summary text
                if len(summary_data) >= 2:
                    st.success(f"""
                    **🚗 Vehicle Identified:** {summary_data[0][1]} {summary_data[1][1] if len(summary_data) > 1 else ''} 
                    {f'({summary_data[2][1]})' if len(summary_data) > 2 else ''}
                    """)
            else:
                st.info("Upload an image to see classification results.")
    
    # Additional information
    st.markdown("---")
    with st.expander("ℹ️ About This System"):
        st.markdown(f"""
        **Advanced Multi-Stage Classification Pipeline:**
        
        **Stage 1 - Car Make Detection:**
        - Uses general YOLO classifier trained on car fronts/rears
        - Identifies manufacturer from {len(labels.get('car_makes', []))} supported brands
        
        **Stage 2 - Specific Model Identification:**
        - Loads brand-specific specialized models
        - Trained separately for each car manufacturer
        - Provides exact model identification with ID numbers
        
        **Stage 3 - Body Type Classification:**
        - Identifies vehicle body style and type
        - Supports {len(labels.get('body_types', []))} different categories
        
        **Supported Classifications:**
        - **Car Makes**: {', '.join(labels.get('car_makes', []))}
        - **Body Types**: {', '.join(labels.get('body_types', []))}
        - **Total Models**: {sum(len(models) for models in model_labels.values())} specific car models
        
        **Data Sources:**
        - Make Names: {'✅ Loaded' if name_mappings['make_names'] else '❌ Not available'}
        - Model Names: {'✅ Loaded' if name_mappings['model_names'] else '❌ Not available'}
        - Model Mappings: ✅ Extracted from dataset structure
        
        **Tips for Best Results:**
        - Use clear, well-lit images
        - Ensure the car is the main subject
        - Front or rear views work best for make identification
        - Side views help with body type classification
        """)

if __name__ == "__main__":
    main()