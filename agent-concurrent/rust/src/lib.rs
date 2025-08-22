use pyo3::prelude::*;
use pyo3::types::PyList;
use rayon::prelude::*;

/// A high-performance system runner that executes system updates in parallel.
///
/// This runner is stateless. When `run` is called, it iterates over the provided
/// systems using a parallel thread pool from Rayon, calling the `update`
/// method on each one.
#[pyclass(name = "ParallelSystemRunner", module = "agent_concurrent_core")]
struct ParallelSystemRunner;

#[pymethods]
impl ParallelSystemRunner {
    /// Creates a new instance of the ParallelSystemRunner.
    #[new]
    fn new() -> Self {
        ParallelSystemRunner
    }

    /// Executes the `update` method on all systems in parallel.
    ///
    /// Args:
    ///     systems (list[object]): The list of system objects to run.
    ///     current_tick (int): The current simulation tick/step.
    ///
    /// Returns:
    ///     None
    fn run(&self, py: Python, systems: &Bound<'_, PyList>, current_tick: i64) -> PyResult<()> {
        // Convert the Python list of systems into a Rust Vec of PyObjects.
        let systems_vec: Vec<PyObject> = systems.iter().map(|obj| obj.to_object(py)).collect();

        // Release the GIL, allowing our Rust code to work in parallel.
        py.allow_threads(|| {
            systems_vec.par_iter().try_for_each(|system| {
                // For each parallel task, re-acquire the GIL to safely interact
                // with the Python object.
                Python::with_gil(|py| {
                    system.call_method1(py, "update", (current_tick,))?;
                    Ok(())
                })
            })
        })
    }
}

/// The Python module definition.
#[pymodule]
fn agent_concurrent_core(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ParallelSystemRunner>()?;
    Ok(())
}