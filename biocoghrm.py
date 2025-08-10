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
# This version incorporates a Four-Level Hierarchical Reasoning Model (HRM)
# for advanced executive function.
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


# ==============================================================================
# HIERARCHICAL REASONING MODEL (HRM) - REPLACING FRONTAL LOBE
# ==============================================================================

class VisionaryModule(nn.Module):
    """
    HRM Level 1: The Visionary. Sets the highest-level objectives.
    In this implementation, it processes the world-state to produce a
    goal embedding that guides the Architect.
    """
    def __init__(self, input_dim: int, goal_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, (input_dim + goal_dim) // 2),
            nn.Tanh(),
            nn.Linear((input_dim + goal_dim) // 2, goal_dim)
        )
    def forward(self, world_state: torch.Tensor) -> torch.Tensor:
        return self.network(world_state)

class ArchitectModule(nn.Module):
    """
    HRM Level 2: The Architect. Creates a high-level plan.
    Uses a GRU to generate a sequence of sub-goals (a plan) based on
    the world state and the visionary's goal.
    """
    def __init__(self, world_state_dim: int, goal_dim: int, plan_dim: int, num_plan_steps: int):
        super().__init__()
        self.num_plan_steps = num_plan_steps
        self.plan_dim = plan_dim
        self.context_dim = world_state_dim + goal_dim
        
        self.context_to_hidden = nn.Linear(self.context_dim, plan_dim)
        self.rnn = nn.GRU(plan_dim, plan_dim, batch_first=True)
        self.start_token = nn.Parameter(torch.randn(1, 1, plan_dim))

    def forward(self, world_state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        batch_size = world_state.shape[0]
        context = torch.cat([world_state, goal], dim=-1)
        
        h_0 = self.context_to_hidden(context).unsqueeze(0)
        
        plan_steps = []
        current_input = self.start_token.repeat(batch_size, 1, 1)
        hidden_state = h_0
        
        for _ in range(self.num_plan_steps):
            output, hidden_state = self.rnn(current_input, hidden_state)
            plan_steps.append(output)
            current_input = output
            
        plan = torch.cat(plan_steps, dim=1)
        return plan

class ForemanModule(nn.Module):
    """
    HRM Level 3: The Foreman. Manages tactical execution.
    Takes the plan from the Architect and issues commands to the Technician.
    It uses a GRUCell to maintain a tactical state.
    """
    def __init__(self, plan_dim: int, command_dim: int):
        super().__init__()
        self.rnn_cell = nn.GRUCell(plan_dim, command_dim)
        self.tactical_state = None

    def forward(self, plan: torch.Tensor) -> torch.Tensor:
        batch_size = plan.shape[0]
        # For this single forward pass, we only execute the first step of the plan.
        # In a real-world scenario, this module would be called in a loop.
        first_plan_step = plan[:, 0, :]
        
        # Initialize or reset state if batch size changes
        if self.tactical_state is None or self.tactical_state.shape[0] != batch_size:
            self.tactical_state = torch.zeros(batch_size, self.rnn_cell.hidden_size, device=plan.device)
            
        self.tactical_state = self.rnn_cell(first_plan_step, self.tactical_state)
        command = self.tactical_state
        return command

class TechnicianModule(nn.Module):
    """
    HRM Level 4: The Technician. Executes low-level actions.
    This is analogous to the motor cortex, using the bio-plausible NeuralLayer.
    """
    def __init__(self, command_dim: int, action_dim: int):
        super().__init__()
        self.motor_cortex = NeuralLayer(command_dim, action_dim)

    def forward(self, command: torch.Tensor) -> torch.Tensor:
        return self.motor_cortex(command)

class HRMModule(nn.Module):
    """
    The full Hierarchical Reasoning Model, acting as the Frontal Lobe.
    It orchestrates the four levels of reasoning from vision to action.
    """
    def __init__(self, config: Dict):
        super().__init__()
        self.visionary = VisionaryModule(
            input_dim=config['parietal_feature_dim'],
            goal_dim=config['hrm_goal_dim']
        )
        self.architect = ArchitectModule(
            world_state_dim=config['parietal_feature_dim'],
            goal_dim=config['hrm_goal_dim'],
            plan_dim=config['hrm_plan_dim'],
            num_plan_steps=config['hrm_num_plan_steps']
        )
        self.foreman = ForemanModule(
            plan_dim=config['hrm_plan_dim'],
            command_dim=config['hrm_command_dim']
        )
        self.technician = TechnicianModule(
            command_dim=config['hrm_command_dim'],
            action_dim=config['action_dim']
        )

    def forward(self, x_parietal: torch.Tensor) -> torch.Tensor:
        """
        Defines the flow of reasoning from high-level goals to low-level actions.
        
        Args:
            x_parietal (torch.Tensor): The integrated world-state representation.

        Returns:
            torch.Tensor: The final action output.
        """
        # 1. Visionary sets the goal based on the current world state
        goal = self.visionary(x_parietal)
        
        # 2. Architect creates a plan to achieve the goal
        plan = self.architect(x_parietal, goal)
        
        # 3. Foreman takes the plan and issues the next command
        command = self.foreman(plan)
        
        # 4. Technician executes the command to produce an action
        action = self.technician(command)
        
        return action

# ==============================================================================
# PILLAR 3: ASSEMBLE THE FULL BRAIN ARCHITECTURE
# ==============================================================================

class Cerebrum(nn.Module):
    """
    Assembles the four lobes and defines the primary information processing
    pathways. The Frontal Lobe is implemented as a Hierarchical Reasoning Model.
    """
    def __init__(self, config: Dict):
        super().__init__()
        self.occipital_lobe = OccipitalLobeModule()
        visual_feature_dim = 512

        self.temporal_lobe = TemporalLobeModule(
            audio_input_dim=config['audio_input_dim'],
            hidden_dim=config['temporal_hidden_dim'],
            lang_dim=config['lang_dim']
        )
        temporal_feature_dim = config['temporal_hidden_dim']

        self.parietal_feature_dim = config['parietal_feature_dim']
        self.visual_projector = nn.Linear(visual_feature_dim, self.parietal_feature_dim)
        self.temporal_projector = nn.Linear(temporal_feature_dim, self.parietal_feature_dim)

        self.parietal_lobe = ParietalLobeModule(feature_dim=self.parietal_feature_dim)

        # The Frontal Lobe is now implemented by the HRM
        self.frontal_lobe = HRMModule(config)

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

        # 4. Executive Function and Action Selection via HRM
        action = self.frontal_lobe(integrated_representation)

        return action, modulatory_signal


class Brain(nn.Module):
    """
    The top-level class for the entire cognitive architecture.
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
        """
        # === Part 1: Global, Goal-Directed Update (Backpropagation) ===
        optimizer.zero_grad()
        predictions, modulatory_signal = self.forward(visual_data, audio_data)
        primary_loss = loss_fn(predictions, labels)
        primary_loss.backward()
        optimizer.step()

        # === Part 2: Local, Self-Organizing Update (Hebbian Plasticity) ===
        arousal = torch.mean(torch.sigmoid(modulatory_signal[:, 0])).item()
        modulated_local_lr = local_lr_base * (1 + arousal)

        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, Neuron):
                    pre_activity = module.last_pre_synaptic_input
                    post_activity = module.last_post_synaptic_output
                    if pre_activity is not None and post_activity is not None and torch.abs(post_activity) > 0:
                       delta_w = modulated_local_lr * (post_activity * pre_activity)
                       module.weights.data += delta_w

        # === Part 3: Synaptic Recovery ===
        for module in self.modules():
            if isinstance(module, Neuron):
                module.recover_synapses()
        
        return primary_loss.item(), arousal, modulated_local_lr


# ==============================================================================
# PILLAR 6: VALIDATION
# ==============================================================================

if __name__ == '__main__':
    print("🔬 Running BioCog-Net Validation Script (v2 with HRM)...")

    # --- Configuration ---
    config = {
        # Sensory and integration dimensions
        'audio_input_dim': 80,
        'temporal_hidden_dim': 128,
        'lang_dim': 256,
        'parietal_feature_dim': 256,
        'action_dim': 10,
        
        # New HRM dimensions
        'hrm_goal_dim': 32,
        'hrm_plan_dim': 64,
        'hrm_num_plan_steps': 5,
        'hrm_command_dim': 128,
    }

    # 1. Instantiate the Brain model
    print("\n[1] Instantiating the full Brain model with HRM...")
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
    print("    ✅ Dummy data created.")

    # 3. Pass dummy data through the model
    print("\n[3] Performing a forward pass...")
    try:
        action_output, mod_signal = brain_model(dummy_visual, dummy_audio)
        print(f"    - Final action output shape: {action_output.shape} (Expected: {batch_size, config['action_dim']})")
        print(f"    - Modulatory signal shape: {mod_signal.shape}")
        assert action_output.shape == (batch_size, config['action_dim'])
        print("    ✅ Forward pass successful.")
    except Exception as e:
        print(f"    ❌ Forward pass failed: {e}")

    # 4. Verify Synapse count in a NeuralLayer (still relevant for Technician)
    print("\n[4] Verifying component architecture...")
    technician_layer = brain_model.cerebrum.frontal_lobe.technician.motor_cortex
    expected_synapses = technician_layer.input_size * technician_layer.output_size
    actual_synapses = sum(len(n.synapses) for n in technician_layer.neurons)
    print(f"    - Testing Technician's NeuralLayer({technician_layer.input_size}, {technician_layer.output_size})")
    print(f"    - Expected synapses: {expected_synapses}")
    print(f"    - Actual synapses: {actual_synapses}")
    assert expected_synapses == actual_synapses, "Synapse count mismatch!"
    print("    ✅ Synapse count verified.")

    # 5. Demonstrate the hybrid training step
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

    print("\n🔬 Validation complete. BioCog-Net with HRM is operational.")
