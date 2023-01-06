/* * Neuron submodule * */

/**
    Trait for the implementation of all the Neuron models.
    It represents a general Neuron of a Layer
*/
pub trait Neuron: Send {
    /** The neuron function is invoked only when some input spikes arrive from the previous layer;
        therefore, extra_weighted_sum is always > 0
        - t: time instant when the input spikes arrive
        - extra_weighted_sum: dot product between *input spikes* and incoming *weights*
        - intra_weighted_sum: dot product between the *input spikes of the previous instant in which at
                              least one neuron (of the previous layer) fired* and the *intra-layer weights*

    */
    fn compute_v_mem(&mut self, t: u64, extra_weighted_sum: f64, intra_weighted_sum: f64) -> u8;

    /**
        Bring the Neuron to the initial state: initialize all data structures
     */
    fn initialize(&mut self);
}