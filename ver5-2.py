import torch
import torch.nn as nn
import json
import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

@dataclass
class NeuromodulatorState:
    """Represents the current neuromodulator levels in the brain"""
    dopamine: float = 0.5      # Reward/motivation
    norepinephrine: float = 0.3 # Attention/arousal
    serotonin: float = 0.4     # Mood/learning rate
    acetylcholine: float = 0.3  # Attention/plasticity
    gaba: float = 0.6          # Inhibition/stability

class BiologicalFeedbackSystem:
    """
    Implements biologically-inspired feedback mechanisms for LLM teacher guidance.
    
    This system translates LLM feedback into specific biological signals that
    modulate learning in ways similar to how neurotransmitters work in real brains.
    """
    
    def __init__(self, brain, baseline_neuromodulators: NeuromodulatorState = None):
        self.brain = brain
        self.baseline_modulators = baseline_neuromodulators or NeuromodulatorState()
        self.current_modulators = NeuromodulatorState()
        self.modulation_history = []
        
        # Feedback integration weights
        self.feedback_weights = {
            'performance_error': 0.3,
            'curiosity_drive': 0.2,
            'attention_signal': 0.2,
            'stability_signal': 0.15,
            'reward_prediction_error': 0.15
        }
        
        # Learning from feedback history
        self.feedback_effectiveness = {}
        
    def process_llm_feedback(self, llm_response: Dict[str, Any], 
                           current_performance: Dict[str, float]) -> Dict[str, float]:
        """
        Convert LLM feedback into biological neuromodulator adjustments.
        
        Args:
            llm_response: Response from LLM teacher
            current_performance: Current BioCog-Net performance metrics
            
        Returns:
            Dictionary of neuromodulator adjustments
        """
        modulator_adjustments = {}
        
        # 1. DOPAMINE: Reward Prediction Error Signal
        expected_performance = llm_response.get('expected_performance', 0.5)
        actual_performance = current_performance.get('accuracy', 0.0)
        
        reward_prediction_error = actual_performance - expected_performance
        dopamine_adjustment = torch.sigmoid(torch.tensor(reward_prediction_error * 2.0)).item()
        modulator_adjustments['dopamine'] = dopamine_adjustment
        
        # 2. NOREPINEPHRINE: Attention/Arousal Signal
        attention_keywords = ['focus', 'attention', 'concentrate', 'important']
        attention_score = sum(1 for keyword in attention_keywords 
                            if keyword in str(llm_response).lower()) / len(attention_keywords)
        
        # High loss or low performance increases arousal
        loss_factor = current_performance.get('train_loss', 1.0)
        arousal_signal = attention_score + (loss_factor - 1.0) * 0.5
        norepinephrine_adjustment = torch.clamp(torch.tensor(arousal_signal), 0.0, 1.0).item()
        modulator_adjustments['norepinephrine'] = norepinephrine_adjustment
        
        # 3. ACETYLCHOLINE: Learning Rate Modulation
        learning_keywords = ['learn', 'adapt', 'improve', 'plasticity']
        learning_emphasis = sum(1 for keyword in learning_keywords 
                              if keyword in str(llm_response).lower()) / len(learning_keywords)
        
        # Uncertainty increases ACh (exploration vs exploitation)
        prediction_uncertainty = current_performance.get('prediction_entropy', 0.5)
        acetylcholine_adjustment = (learning_emphasis + prediction_uncertainty) / 2.0
        modulator_adjustments['acetylcholine'] = acetylcholine_adjustment
        
        # 4. SEROTONIN: Mood/Confidence Modulation
        positive_keywords = ['good', 'excellent', 'improvement', 'success', 'progress']
        negative_keywords = ['poor', 'decline', 'problem', 'issue', 'concern']
        
        positive_count = sum(1 for keyword in positive_keywords 
                           if keyword in str(llm_response).lower())
        negative_count = sum(1 for keyword in negative_keywords 
                           if keyword in str(llm_response).lower())
        
        mood_score = (positive_count - negative_count) / max(positive_count + negative_count, 1)
        serotonin_adjustment = torch.sigmoid(torch.tensor(mood_score)).item()
        modulator_adjustments['serotonin'] = serotonin_adjustment
        
        # 5. GABA: Stability/Inhibition Signal
        stability_keywords = ['stable', 'consistent', 'reliable', 'steady']
        instability_keywords = ['chaotic', 'unstable', 'erratic', 'inconsistent']
        
        stability_score = sum(1 for keyword in stability_keywords 
                            if keyword in str(llm_response).lower())
        instability_score = sum(1 for keyword in instability_keywords 
                              if keyword in str(llm_response).lower())
        
        # High variance in performance reduces GABA
        performance_variance = current_performance.get('performance_variance', 0.1)
        gaba_adjustment = (stability_score - instability_score - performance_variance * 5.0)
        gaba_adjustment = torch.sigmoid(torch.tensor(gaba_adjustment)).item()
        modulator_adjustments['gaba'] = gaba_adjustment
        
        return modulator_adjustments
    
    def apply_neuromodulation(self, modulator_adjustments: Dict[str, float]) -> None:
        """
        Apply neuromodulator changes to the brain's synapses and neurons.
        
        Args:
            modulator_adjustments: Dictionary of neuromodulator level changes
        """
        # Update current modulator levels
        for modulator, adjustment in modulator_adjustments.items():
            current_level = getattr(self.current_modulators, modulator)
            new_level = current_level + adjustment * 0.1  # Gradual changes
            new_level = torch.clamp(torch.tensor(new_level), 0.0, 1.0).item()
            setattr(self.current_modulators, modulator, new_level)
        
        # Apply to synapses
        all_synapses = self.brain.get_all_synapses()
        
        for synapse in all_synapses:
            # Dopamine: Modulates learning rate and reward sensitivity
            dopamine_effect = self.current_modulators.dopamine
            synapse.reward_sensitivity = dopamine_effect
            
            # Norepinephrine: Modulates arousal and attention
            ne_effect = self.current_modulators.norepinephrine
            synapse.metaplasticity_state *= (1.0 + ne_effect * 0.2)
            
            # Acetylcholine: Modulates plasticity and learning rate
            ach_effect = self.current_modulators.acetylcholine
            if hasattr(synapse, 'learning_rate'):
                synapse.learning_rate *= (1.0 + ach_effect * 0.3)
            
            # Serotonin: Modulates overall learning confidence
            serotonin_effect = self.current_modulators.serotonin
            synapse.confidence_modulator = serotonin_effect
            
            # GABA: Modulates inhibition and stability
            gaba_effect = self.current_modulators.gaba
            if hasattr(synapse, 'inhibitory_strength'):
                synapse.inhibitory_strength = gaba_effect
        
        # Apply to neural layers
        neural_layers = self.brain.get_all_neural_layers()
        
        for layer in neural_layers:
            # Adjust homeostatic set points based on serotonin
            layer.homeostatic_set_point *= (1.0 + self.current_modulators.serotonin * 0.1)
            
            # Adjust noise levels based on norepinephrine
            if hasattr(layer, 'noise_level'):
                layer.noise_level = self.current_modulators.norepinephrine * 0.05
    
    def generate_curiosity_drive(self, current_state: Dict[str, Any]) -> float:
        """
        Generate intrinsic motivation signal based on prediction uncertainty.
        
        This implements curiosity-driven learning where the brain seeks out
        experiences that maximize learning (similar to dopamine in exploration).
        """
        # Factors that increase curiosity
        prediction_uncertainty = current_state.get('prediction_entropy', 0.5)
        novelty_score = current_state.get('novelty_detection', 0.5)
        learning_progress = current_state.get('recent_improvement', 0.0)
        
        # Curiosity is high when uncertainty is high but progress is being made
        curiosity = prediction_uncertainty * (1.0 + learning_progress) * novelty_score
        
        return torch.clamp(torch.tensor(curiosity), 0.0, 1.0).item()
    
    def compute_attention_signal(self, llm_guidance: Dict[str, Any], 
                               brain_state: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute attention allocation across different brain modules.
        
        Args:
            llm_guidance: Guidance from LLM teacher
            brain_state: Current state of brain modules
            
        Returns:
            Attention weights for different brain regions
        """
        attention_weights = {}
        
        # Default equal attention
        brain_modules = ['frontal', 'parietal', 'temporal', 'occipital']
        base_attention = 1.0 / len(brain_modules)
        
        for module in brain_modules:
            attention_weights[module] = base_attention
        
        # Adjust based on LLM guidance
        focus_areas = llm_guidance.get('focus_areas', [])
        
        for area in focus_areas:
            if area in attention_weights:
                attention_weights[area] *= 2.0  # Double attention to focus areas
        
        # Normalize
        total_attention = sum(attention_weights.values())
        attention_weights = {k: v/total_attention for k, v in attention_weights.items()}
        
        return attention_weights
    
    def implement_error_correction_feedback(self, prediction_errors: torch.Tensor,
                                          target_patterns: torch.Tensor) -> Dict[str, Any]:
        """
        Implement error correction feedback similar to cerebellum.
        
        Args:
            prediction_errors: Current prediction errors
            target_patterns: Desired output patterns
            
        Returns:
            Error correction signals for different pathways
        """
        batch_size = prediction_errors.shape[0]
        
        # Compute error magnitudes per sample
        error_magnitudes = torch.norm(prediction_errors, dim=-1)
        
        # Generate correction signals
        correction_signals = {}
        
        # Forward model correction (cerebellar-like)
        correction_signals['forward_model'] = {
            'error_magnitude': error_magnitudes.mean().item(),
            'correction_direction': (target_patterns - prediction_errors).mean(dim=0).tolist(),
            'learning_rate_multiplier': torch.clamp(error_magnitudes.mean() * 2.0, 0.1, 3.0).item()
        }
        
        # Inverse model correction (motor control)
        correction_signals['inverse_model'] = {
            'motor_adjustment': torch.sign(prediction_errors).mean(dim=0).tolist(),
            'confidence_adjustment': (1.0 - error_magnitudes.mean()).item()
        }
        
        return correction_signals
    
    def adaptive_feedback_integration(self, multiple_feedbacks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Integrate feedback from multiple LLM teachers with adaptive weighting.
        
        Args:
            multiple_feedbacks: List of feedback from different LLM teachers
            
        Returns:
            Integrated feedback with confidence weights
        """
        if not multiple_feedbacks:
            return {}
        
        # Track which teachers give better advice
        teacher_effectiveness = {}
        
        integrated_feedback = {}
        
        # For each feedback type, compute weighted average
        all_keys = set()
        for feedback in multiple_feedbacks:
            all_keys.update(feedback.keys())
        
        for key in all_keys:
            values = []
            weights = []
            
            for i, feedback in enumerate(multiple_feedbacks):
                if key in feedback:
                    values.append(feedback[key])
                    # Weight by historical effectiveness of this teacher
                    teacher_weight = teacher_effectiveness.get(f'teacher_{i}', 1.0)
                    weights.append(teacher_weight)
            
            if values and weights:
                # Weighted average
                if isinstance(values[0], (int, float)):
                    weighted_avg = sum(v * w for v, w in zip(values, weights)) / sum(weights)
                    integrated_feedback[key] = weighted_avg
                else:
                    # For non-numeric values, use majority vote with weights
                    integrated_feedback[key] = max(values, key=lambda x: weights[values.index(x)])
        
        return integrated_feedback
    
    def generate_feedback_report(self) -> str:
        """Generate detailed report of current neuromodulator states and their effects."""
        
        report = f"""
        BioCog-Net Neuromodulator Status Report
        ======================================
        
        Current Levels:
        - Dopamine (Reward/Motivation): {self.current_modulators.dopamine:.3f}
        - Norepinephrine (Attention/Arousal): {self.current_modulators.norepinephrine:.3f}
        - Serotonin (Mood/Confidence): {self.current_modulators.serotonin:.3f}
        - Acetylcholine (Learning/Plasticity): {self.current_modulators.acetylcholine:.3f}
        - GABA (Stability/Inhibition): {self.current_modulators.gaba:.3f}
        
        Effects on Learning:
        - Learning Rate Modulation: {self.current_modulators.acetylcholine * 0.3:.3f}
        - Attention Focus: {self.current_modulators.norepinephrine:.3f}
        - Exploration Drive: {self.current_modulators.dopamine:.3f}
        - Learning Confidence: {self.current_modulators.serotonin:.3f}
        - Network Stability: {self.current_modulators.gaba:.3f}
        
        Recommendations:
        """
        
        # Add recommendations based on current levels
        if self.current_modulators.dopamine < 0.3:
            report += "\n- Consider reward/achievement-based tasks to boost motivation"
        if self.current_modulators.acetylcholine < 0.3:
            report += "\n- Increase task novelty to boost learning drive"
        if self.current_modulators.gaba < 0.3:
            report += "\n- Reduce task difficulty to improve stability"
            
        return report

class EnhancedLLMTrainer:
    """
    Enhanced LLM trainer with biological feedback integration.
    """
    
    def __init__(self, brain, api_keys: Dict[str, str], preferred_teacher: str = "gemini"):
        # ... (existing initialization code)
        self.biological_feedback = BiologicalFeedbackSystem(brain)
        self.feedback_history = []
        
    async def biological_feedback_step(self, env, training_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform training step with biological feedback integration.
        
        Args:
            env: Continuous learning environment
            training_stats: Current training statistics
            
        Returns:
            Complete feedback and neuromodulation results
        """
        # 1. Get LLM feedback
        llm_feedback = await self.real_time_feedback_loop(env, training_stats)
        
        # 2. Process into biological signals
        neuromodulator_adjustments = self.biological_feedback.process_llm_feedback(
            llm_feedback, training_stats
        )
        
        # 3. Apply neuromodulation to brain
        self.biological_feedback.apply_neuromodulation(neuromodulator_adjustments)
        
        # 4. Generate curiosity-driven exploration
        curiosity_drive = self.biological_feedback.generate_curiosity_drive(training_stats)
        
        # 5. Compute attention allocation
        attention_weights = self.biological_feedback.compute_attention_signal(
            llm_feedback, training_stats
        )
        
        # 6. Track feedback effectiveness
        feedback_record = {
            'llm_feedback': llm_feedback,
            'neuromodulator_adjustments': neuromodulator_adjustments,
            'curiosity_drive': curiosity_drive,
            'attention_weights': attention_weights,
            'performance_after': None  # Will be filled in next step
        }
        
        self.feedback_history.append(feedback_record)
        
        return feedback_record
    
    def evaluate_feedback_effectiveness(self) -> Dict[str, float]:
        """
        Analyze which types of feedback lead to better learning outcomes.
        
        Returns:
            Effectiveness scores for different feedback mechanisms
        """
        if len(self.feedback_history) < 10:
            return {}
        
        effectiveness = {}
        
        # Analyze correlation between feedback types and subsequent performance
        for i in range(len(self.feedback_history) - 5):
            feedback = self.feedback_history[i]
            future_performance = [h.get('performance_after', {}).get('accuracy', 0) 
                                for h in self.feedback_history[i+1:i+6]]
            
            if future_performance:
                avg_future_perf = sum(future_performance) / len(future_performance)
                
                # Correlate with different feedback components
                for component, value in feedback['neuromodulator_adjustments'].items():
                    if component not in effectiveness:
                        effectiveness[component] = []
                    effectiveness[component].append((value, avg_future_perf))
        
        # Compute correlations
        final_effectiveness = {}
        for component, pairs in effectiveness.items():
            if len(pairs) > 3:
                values, performances = zip(*pairs)
                correlation = np.corrcoef(values, performances)[0, 1]
                final_effectiveness[component] = correlation
        
        return final_effectiveness

# Example usage and integration
def create_enhanced_biocog_system():
    """Create BioCog-Net system with advanced biological feedback."""
    
    # Initialize components
    brain = Brain()  # Your existing Brain class
    env = ContinuousLearningEnvironment(brain, capacity=5000)
    
    # API keys for LLM teachers
    api_keys = {
        "gemini": "your-gemini-key",
        "mistral": "your-mistral-key", 
        "groq": "your-groq-key"
    }
    
    # Enhanced trainer with biological feedback
    trainer = EnhancedLLMTrainer(brain, api_keys, preferred_teacher="gemini")
    
    return brain, env, trainer

async def run_biological_feedback_training():
    """Demonstrate biological feedback training loop."""
    
    brain, env, trainer = create_enhanced_biocog_system()
    
    print("🧠 Starting BioCog-Net with Biological Feedback Training")
    print("=" * 60)
    
    for step in range(50):
        # Generate training data
        visual_data = torch.randn(8, 3, 224, 224)
        auditory_data = torch.randn(8, 80)
        labels = torch.randint(0, 10, (8,))
        
        # Training step with performance metrics
        training_stats = env.continuous_learning_step(visual_data, auditory_data, labels)
        
        # Biological feedback integration
        feedback_result = await trainer.biological_feedback_step(env, training_stats)
        
        # Update feedback effectiveness tracking
        if step > 0:
            trainer.feedback_history[-2]['performance_after'] = training_stats
        
        # Periodic reporting
        if step % 10 == 0:
            print(f"\nStep {step}:")
            print(f"  Accuracy: {training_stats['accuracy']:.4f}")
            print(f"  Loss: {training_stats['train_loss']:.4f}")
            print(f"  Curiosity Drive: {feedback_result['curiosity_drive']:.3f}")
            print(f"  Dopamine Level: {trainer.biological_feedback.current_modulators.dopamine:.3f}")
            print(f"  Attention Focus: {max(feedback_result['attention_weights'], key=feedback_result['attention_weights'].get)}")
            
            # Print neuromodulator report every 20 steps
            if step % 20 == 0:
                print("\n" + trainer.biological_feedback.generate_feedback_report())
    
    # Final effectiveness analysis
    effectiveness = trainer.evaluate_feedback_effectiveness()
    print(f"\n📊 Feedback Effectiveness Analysis:")
    for component, score in effectiveness.items():
        print(f"  {component}: {score:.3f}")
    
    print("\n✅ Biological Feedback Training Complete!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_biological_feedback_training())
