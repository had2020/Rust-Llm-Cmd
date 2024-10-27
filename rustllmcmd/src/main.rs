use tensorflow::{Graph, Session, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let graph = Graph::new()?;
    let session = Session::new(&graph)?;
    session.run(&[
        Tensor::new(&[1, MAX_SEQUENCE_LENGTH])?,
    ])?;
    let output_tensor = session.run(&[output_tensor_index])?[0];
    //todo add processing
    Ok(())
}