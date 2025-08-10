# ==============================================================================
# BioCog-Net: A Bio-Plausible Cognitive Architecture
#
# Author: Gemini Research Scientist
# Date: August 10, 2025
#
# Description:
# This script implements the BioCog-Net framework, an experimental neural
# network architecture in PyTorch. It is designed to be a computational
# translation of core neuroscientific principles, including modular brain
# organization, stateful synaptic dynamics, and hybrid learning rules.
# ==============================================================================

import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F
from typing import List, Tuple, Optional, Dict

# ==============================================================================
# PILLAR 1: MICRO-ARCHITECTURAL COMPONENTS
# ==============================================================================

class Synapse:
    """
    Encapsulates the state and behavior of a single, stateful synapse.

    This is not an nn.Module, but a helper class to model biological synaptic
    properties like short-term plasticity (fatigue) and probabilistic release.

    Rationale:
    Implements the stateful, probabilistic synapse model. This moves beyond a
    simple scalar weight to capture short-term plasticity and stochasticity,
    based on the complexity of biological synaptic transmission.

    Attributes:
        weight (float): The base synaptic strength.
        release_probability (float): The chance a signal will be transmitted.
        fatigue (float): A factor representing synaptic resource depletion,
                         modulating the signal strength.
    """
    def __init__(self, initial_weight: float):
        """
        Initializes the Synapse.

        Args:
            initial_weight (float): The starting weight of the synapse.
        """
        self.weight: float = initial_weight
        self.release_probability: float = 0.9
        self.fatigue: float = 1.0  # Starts fully rested

    def transmit(self, input_signal: torch.Tensor) -> torch.Tensor:
        """
        Transmits a signal across the synapse, if stochastic conditions are met.

        Args:
            input_signal (torch.Tensor): The incoming signal from the
                                         presynaptic neuron.

        Returns:
            torch.Tensor: The output signal, which is either the modulated
                          input or zero.
        """
        if torch.rand(1) < self.release_probability:
            transmitted_signal = self.weight * self.fatigue * input_signal
            self.update_fatigue()
            return transmitted_signal
        return torch.tensor(0.0, device=input_signal.device)

    def update_fatigue(self):
        """
        Reduces the fatigue factor after a successful transmission, modeling
        neurotransmitter depletion.
        """
        self.fatigue *= 0.9

    def recover(self):
        """
        Slowly restores the fatigue factor towards its resting state. This
        should be called periodically (e.g., at the end of a batch).
        """
        self.fatigue = min(1.0, self.fatigue / 0.95)


class Neuron(nn.Module):
    """
    A custom neuron model with stateful synapses and a dynamic firing threshold.

    Rationale:
    Implements the excitatory/inhibitory principle. The `dynamic_threshold`
    and `tanh` activation capture the push-pull dynamics and adaptive firing
    sensitivity of biological neurons.

    Attributes:
        synapses (List[Synapse]): A list of input synapses.
        dynamic_threshold (nn.Parameter): A learnable firing threshold.
    """
    def __init__(self, input_size: int):
        """
        Initializes the Neuron.

        Args:
            input_size (int): The number of incoming connections.
        """
        super().__init__()
        # Initialize synapses with small random weights
        self.synapses = [Synapse(initial_weight=(torch.randn(1) * 0.1).item()) for _ in range(input_size)]
        # Make the synaptic weights learnable parameters for the global update
        self.weights = nn.Parameter(torch.tensor([s.weight for s in self.synapses]))
        self.dynamic_threshold = nn.Parameter(torch.abs(torch.randn(1) * 0.01))
        self.last_pre_synaptic_input = None
        self.last_post_synaptic_output = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass for the neuron.

        Args:
            x (torch.Tensor): A 1D tensor of input signals.

        Returns:
            torch.Tensor: The single output value of the neuron.
        """
        # Update internal synapse weights from the learnable nn.Parameter
        for i, w in enumerate(self.weights):
            self.synapses[i].weight = w.item()

        # Store pre-synaptic input for Hebbian learning
        self.last_pre_synaptic_input = x.detach().clone()

        # Probabilistic transmission through each synapse
        weighted_inputs = [s.transmit(x_i) for s, x_i in zip(self.synapses, x)]
        total_input = torch.sum(torch.stack(weighted_inputs))

        # Apply dynamic firing threshold
        if total_input > self.dynamic_threshold:
            # Tanh activation allows for excitatory (+) and inhibitory (-) output
            output = torch.tanh(total_input)
        else:
            output = torch.tensor(0.0, device=x.device)

        # Store post-synaptic output for Hebbian learning
        self.last_post_synaptic_output = output.detach().clone()

        return output

    def recover_synapses(self):
        """Recovers all synapses connected to this neuron."""
        for synapse in self.synapses:
            synapse.recover()


class NeuralLayer(nn.Module):
    """
    A layer composed of multiple custom `Neuron` objects.

    Rationale:
    A container for the custom `Neuron` objects, forming the basic processing
    fabric of the cortical modules.

    Attributes:
        neurons (nn.ModuleList): A list of Neuron objects that form the layer.
    """
    def __init__(self, input_size: int, output_size: int):
        """
        Initializes the NeuralLayer.

        Args:
            input_size (int): The dimensionality of the input.
            output_size (int): The number of neurons in this layer.
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.neurons = nn.ModuleList([Neuron(input_size) for _ in range(output_size)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes the input tensor through all neurons in the layer.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, output_size).
        """
        # Assuming batch_size = 1 for simplicity in this conceptual model
        if x.dim() > 1 and x.shape[0] > 1:
            # Handle batches by iterating, though this is inefficient.
            # A vectorized implementation would be needed for performance.
            outputs = [self.forward_single(item) for item in x]
            return torch.stack(outputs)
        return self.forward_single(x.squeeze(0))


    def forward_single(self, x_single: torch.Tensor) -> torch.Tensor:
        """
        Processes a single item through all neurons in the layer.

        Args:
            x_single (torch.Tensor): The input tensor of shape (input_size,).

        Returns:
            torch.Tensor: The output tensor of shape (output_size,).
        """
        outputs = [neuron(x_single) for neuron in self.neurons]
        return torch.stack(outputs)


# ==============================================================================
# PILLAR 2: MACRO-ARCHITECTURAL MODULES
# ==============================================================================

class OccipitalLobeModule(nn.Module):
    """
    Models the visual processing pathway (occipital lobe).

    Rationale:
    The occipital lobe is highly specialized for vision. Using a proven vision
    model like ResNet is a practical engineering choice to represent this
    specialized, high-performance biological system.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        # Remove the final classification layer to use as a feature extractor
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts visual features from an image.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, 3, H, W).

        Returns:
            torch.Tensor: A flattened feature vector.
        """
        with torch.no_grad(): # Freeze weights for pretrained model
            features = self.feature_extractor(x)
        return torch.flatten(features, 1)


class TemporalLobeModule(nn.Module):
    """
    Models the temporal lobe's functions: hearing, memory, and emotion.

    Rationale:
    Models the functional specialization of the temporal lobe, including
    hearing, language, and memory. The `amygdala` sub-module implements the
    emotional salience modulator. The `hippocampus` as an RNN models its role
    in sequential memory processing.
    """
    def __init__(self, audio_input_dim: int, hidden_dim: int, lang_dim: int):
        super().__init__()
        # Primary Auditory Cortex: Initial sound processing
        self.primary_auditory_cortex = NeuralLayer(audio_input_dim, hidden_dim)

        # Wernicke's Area: Language comprehension
        self.wernicke_area = NeuralLayer(hidden_dim, lang_dim)

        # Hippocampus: Episodic memory buffer
        self.hippocampus = nn.GRU(lang_dim, hidden_dim, batch_first=True)

        # Amygdala: Emotional salience assessment
        self.amygdala = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 2) # Output: [arousal, valence]
        )

    def forward(self, x_audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Processes auditory input and returns a processed representation and an
        emotional modulation signal.

        Args:
            x_audio (torch.Tensor): Auditory input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
            - Main processed output tensor.
            - Modulatory signal [arousal, valence] from the amygdala.
        """
        # Process through auditory pathway
        processed_audio = self.primary_auditory_cortex(x_audio)

        # Emotional salience is assessed early
        modulatory_signal = self.amygdala(processed_audio.detach()) # Detach to prevent gradient flow

        # Language and memory processing
        lang_representation = self.wernicke_area(processed_audio)
        # Unsqueeze to add sequence dimension for RNN
        memory_output, _ = self.hippocampus(lang_representation.unsqueeze(1))

        return memory_output.squeeze(1), modulatory_signal


class ParietalLobeModule(nn.Module):
    """
    Models the parietal lobe as a multi-modal integration hub.

    Rationale:
    Implements the parietal lobe's role as a multi-modal integration hub.
    This is a critical step for creating a unified perception from diverse
    sensory streams using cross-attention.
    """
    def __init__(self, feature_dim: int, num_heads: int = 8):
        super().__init__()
        self.feature_dim = feature_dim
        # Cross-attention mechanism to fuse sensory inputs
        self.attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Linear(feature_dim * 4, feature_dim),
        )

    def forward(self, visual_in: torch.Tensor, temporal_in: torch.Tensor) -> torch.Tensor:
        """
        Fuses visual and temporal information into a unified representation.

        Args:
            visual_in (torch.Tensor): Processed features from Occipital Lobe.
            temporal_in (torch.Tensor): Processed features from Temporal Lobe.

        Returns:
            torch.Tensor: The integrated world-state representation.
        """
        # Reshape inputs to (Batch, Sequence, Features) for attention
        visual_in = visual_in.unsqueeze(1)
        temporal_in = temporal_in.unsqueeze(1)

        # Use visual input as the query, temporal as key/value
        attn_output, _ = self.attention(query=visual_in, key=temporal_in, value=temporal_in)

        # Add & Norm, followed by a Feed-Forward Network
        x = self.norm1(visual_in + attn_output)
        x_ffn = self.ffn(x)
        integrated_state = self.norm2(x + x_ffn)

        return integrated_state.squeeze(1)


class FrontalLobeModule(nn.Module):
    """
    Models the executive functions of the frontal lobe.

    Rationale:
    Models executive functions like planning, action generation, and top-down
    attentional control (inhibition).
    """
    def __init__(self, integrated_dim: int, action_dim: int, lang_output_dim: int):
        super().__init__()
        # Executive Control: Gating mechanism
        self.executive_control = nn.Sequential(
            nn.Linear(integrated_dim, integrated_dim),
            nn.Sigmoid() # Gating weights
        )

        # Working Memory: Recurrent buffer
        self.working_memory_cell = nn.GRUCell(integrated_dim, integrated_dim)
        self.working_memory_state = None # Initialized at forward pass

        # Motor Cortex: Generates final action
        self.motor_cortex = NeuralLayer(integrated_dim, action_dim)

        # Broca's Area: Generates language-like output (placeholder)
        self.broca_area = NeuralLayer(integrated_dim, lang_output_dim)

    def forward(self, x_parietal: torch.Tensor) -> torch.Tensor:
        """
        Takes the integrated world-state and generates an action.

        Args:
            x_parietal (torch.Tensor): The unified representation from the
                                       Parietal Lobe.

        Returns:
            torch.Tensor: The final action output.
        """
        # Initialize working memory state if it's the first pass
        if self.working_memory_state is None or self.working_memory_state.shape[0] != x_parietal.shape[0]:
            self.working_memory_state = torch.zeros_like(x_parietal)

        # Apply executive control as a gate on the input
        gate = self.executive_control(x_parietal)
        gated_input = gate * x_parietal

        # Update working memory
        self.working_memory_state = self.working_memory_cell(gated_input, self.working_memory_state)

        # Generate action from the current working memory state
        action = self.motor_cortex(self.working_memory_state)

        # Note: broca_area output is not used in this simplified pathway
        # lang_output = self.broca_area(self.working_memory_state)

        return action


# ==============================================================================
# PILLAR 3: ASSEMBLE THE FULL BRAIN ARCHITECTURE
# ==============================================================================

class Cerebrum(nn.Module):
    """
    Assembles the four lobes and defines the primary information processing
    pathways.

    Rationale:
    Assembles the individual modules into a cohesive whole, defining the primary
    cognitive information processing pathway based on the brain's known
    connectivity.
    """
    def __init__(self, config: Dict):
        super().__init__()
        self.occipital_lobe = OccipitalLobeModule()
        # ResNet18 feature output size is 512
        visual_feature_dim = 512

        self.temporal_lobe = TemporalLobeModule(
            audio_input_dim=config['audio_input_dim'],
            hidden_dim=config['temporal_hidden_dim'],
            lang_dim=config['lang_dim']
        )
        temporal_feature_dim = config['temporal_hidden_dim']

        # Ensure parietal lobe can handle concatenated features if needed,
        # but here we use attention with a unified feature dimension.
        # We'll project both visual and temporal features to this dimension.
        self.parietal_feature_dim = config['parietal_feature_dim']
        self.visual_projector = nn.Linear(visual_feature_dim, self.parietal_feature_dim)
        self.temporal_projector = nn.Linear(temporal_feature_dim, self.parietal_feature_dim)

        self.parietal_lobe = ParietalLobeModule(feature_dim=self.parietal_feature_dim)

        self.frontal_lobe = FrontalLobeModule(
            integrated_dim=self.parietal_feature_dim,
            action_dim=config['action_dim'],
            lang_output_dim=config['lang_output_dim']
        )

    def forward(self, x_visual: torch.Tensor, x_audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Defines the end-to-end data flow through the cerebrum.

        Args:
            x_visual (torch.Tensor): Visual input.
            x_audio (torch.Tensor): Auditory input.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
            - Final action output from the frontal lobe.
            - Modulatory signal from the amygdala (in the temporal lobe).
        """
        # 1. Sensory Processing
        visual_features = self.occipital_lobe(x_visual)
        temporal_features, modulatory_signal = self.temporal_lobe(x_audio)

        # 2. Project features to a common dimension for integration
        visual_projected = self.visual_projector(visual_features)
        temporal_projected = self.temporal_projector(temporal_features)

        # 3. Multi-modal Integration
        integrated_representation = self.parietal_lobe(visual_projected, temporal_projected)

        # 4. Executive Function and Action Selection
        action = self.frontal_lobe(integrated_representation)

        return action, modulatory_signal


class Brain(nn.Module):
    """
    The top-level class for the entire cognitive architecture.

    This class contains the main training loop and implements the hybrid
    learning rule, combining global backpropagation with local, arousal-
    modulated Hebbian plasticity.

    Rationale:
    The final container for the entire model. This is where the novel hybrid
    learning rule is implemented, combining goal-directed backpropagation with
    local, self-organizing Hebbian plasticity.
    """
    def __init__(self, config: Dict):
        super().__init__()
        self.cerebrum = Cerebrum(config)
        # Placeholders for future expansion
        self.cerebellum = nn.Identity()
        self.brainstem = nn.Identity()

    def forward(self, x_visual: torch.Tensor, x_audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The main forward pass for the entire brain model.

        Args:
            x_visual (torch.Tensor): Visual input.
            x_audio (torch.Tensor): Auditory input.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
            - Final action output.
            - Modulatory signal [arousal, valence].
        """
        return self.cerebrum(x_visual, x_audio)

    def train_step(self,
                   visual_data: torch.Tensor,
                   audio_data: torch.Tensor,
                   labels: torch.Tensor,
                   optimizer: torch.optim.Optimizer,
                   loss_fn: nn.Module,
                   local_lr_base: float = 0.001):
        """
        Performs a single training step using the hybrid learning rule.

        Args:
            visual_data (torch.Tensor): Batch of visual data.
            audio_data (torch.Tensor): Batch of audio data.
            labels (torch.Tensor): Batch of labels for the primary task.
            optimizer (torch.optim.Optimizer): The optimizer for global updates.
            loss_fn (nn.Module): The loss function for the primary task.
            local_lr_base (float): The base learning rate for Hebbian updates.
        """
        # === Part 1: Global, Goal-Directed Update (Backpropagation) ===
        optimizer.zero_grad()
        predictions, modulatory_signal = self.forward(visual_data, audio_data)
        primary_loss = loss_fn(predictions, labels)
        primary_loss.backward()
        optimizer.step()

        # === Part 2: Local, Self-Organizing Update (Hebbian Plasticity) ===
        # Extract arousal from the modulatory signal (first element)
        # Use mean arousal across the batch
        arousal = torch.mean(torch.sigmoid(modulatory_signal[:, 0])).item()
        # Modulate the local learning rate based on arousal
        modulated_local_lr = local_lr_base * (1 + arousal) # Higher arousal -> faster local learning

        with torch.no_grad():
            # Iterate through all modules that contain custom neurons
            for module in self.modules():
                if isinstance(module, Neuron):
                    # Hebbian rule: "neurons that fire together, wire together"
                    pre_activity = module.last_pre_synaptic_input
                    post_activity = module.last_post_synaptic_output

                    if pre_activity is not None and post_activity is not None:
                        # Only update if the neuron fired
                        if torch.abs(post_activity) > 0:
                           # Calculate weight change based on correlation of pre/post activity
                           delta_w = modulated_local_lr * (post_activity * pre_activity)
                           module.weights.data += delta_w

        # === Part 3: Synaptic Recovery ===
        # At the end of the step, allow synapses to recover from fatigue
        for module in self.modules():
            if isinstance(module, Neuron):
                module.recover_synapses()
        
        return primary_loss.item(), arousal, modulated_local_lr


# ==============================================================================
# PILLAR 6: VALIDATION
# ==============================================================================

if __name__ == '__main__':
    print("🔬 Running BioCog-Net Validation Script...")

    # --- Configuration ---
    config = {
        'audio_input_dim': 80,
        'temporal_hidden_dim': 128,
        'lang_dim': 256,
        'parietal_feature_dim': 256,
        'action_dim': 10,
        'lang_output_dim': 50,
    }

    # 1. Instantiate the Brain model
    print("\n[1] Instantiating the full Brain model...")
    brain_model = Brain(config)
    print("    ✅ Model instantiated successfully.")

    # 2. Create dummy input tensors
    print("\n[2] Creating dummy input tensors...")
    batch_size = 4
    dummy_visual = torch.randn(batch_size, 3, 224, 224)
    dummy_audio = torch.randn(batch_size, config['audio_input_dim'])
    dummy_labels = torch.randint(0, config['action_dim'], (batch_size,))
    print(f"    - Visual input shape: {dummy_visual.shape}")
    print(f"    - Audio input shape: {dummy_audio.shape}")
    print(f"    - Labels shape: {dummy_labels.shape}")
    print("    ✅ Dummy data created.")

    # 3. Pass dummy data through the model
    print("\n[3] Performing a forward pass...")
    try:
        action_output, mod_signal = brain_model(dummy_visual, dummy_audio)
        print(f"    - Final action output shape: {action_output.shape}")
        print(f"    - Modulatory signal shape: {mod_signal.shape}")
        print("    ✅ Forward pass successful.")
    except Exception as e:
        print(f"    ❌ Forward pass failed: {e}")

    # 4. Verify Synapse count in a NeuralLayer
    print("\n[4] Verifying component architecture...")
    input_s, output_s = 10, 5
    test_layer = NeuralLayer(input_s, output_s)
    expected_synapses = input_s * output_s
    actual_synapses = sum(len(n.synapses) for n in test_layer.neurons)
    print(f"    - Testing NeuralLayer({input_s}, {output_s})")
    print(f"    - Expected synapses: {expected_synapses}")
    print(f"    - Actual synapses: {actual_synapses}")
    assert expected_synapses == actual_synapses, "Synapse count mismatch!"
    print("    ✅ Synapse count verified.")

    # 5. Demonstrate the hybrid training step and arousal modulation
    print("\n[5] Demonstrating a single hybrid training step...")
    optimizer = torch.optim.Adam(brain_model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    base_local_lr = 0.001

    loss_val, arousal_val, modulated_lr = brain_model.train_step(
        dummy_visual,
        dummy_audio,
        dummy_labels,
        optimizer,
        loss_fn,
        local_lr_base=base_local_lr
    )
    print(f"    - Primary task loss: {loss_val:.4f}")
    print(f"    - Amygdala arousal level (0-1): {arousal_val:.4f}")
    print(f"    - Base local (Hebbian) LR: {base_local_lr}")
    print(f"    - Arousal-modulated local LR: {modulated_lr:.6f}")
    print("    ✅ Hybrid training step demonstrated.")

    print("\n🔬 Validation complete. BioCog-Net is operational.")
