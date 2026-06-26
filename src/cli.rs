use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use rsomics_common::{RsomicsError, ToolMeta, run};
use serde::Serialize;

use rsomics_boxcox::{
    NormmaxMethod, boxcox, boxcox_llf, boxcox_normmax, check_positive_nonconstant, parse_values,
};

pub const META: ToolMeta = ToolMeta {
    name: env!("CARGO_PKG_NAME"),
    version: env!("CARGO_PKG_VERSION"),
};

/// Box-Cox power transform, log-likelihood, and optimal lambda — value-exact to
/// `scipy.stats.boxcox` / `boxcox_llf` / `boxcox_normmax`.
///
/// With no `--lambda`, the MLE lambda is found by maximizing the log-likelihood
/// (or, with `--normmax-method pearsonr`, the probability-plot correlation) and
/// the data is transformed with it; the chosen lambda is printed as a `# lambda`
/// header. With `--lambda L` the data is transformed for that fixed L. With
/// `--llf-at L` only the log-likelihood at L is printed.
#[derive(Parser, Debug)]
#[command(name = "rsomics-boxcox", version, about, long_about = None)]
pub struct Cli {
    /// Input value TSV; one positive observation per line (whitespace also
    /// separates). `-` is not accepted; pass a path.
    #[arg(value_name = "DATA", required = true)]
    pub data: PathBuf,

    /// Transform with this fixed lambda instead of finding the optimum.
    #[arg(long, value_name = "L", allow_hyphen_values = true)]
    pub lambda: Option<f64>,

    /// Print only the Box-Cox log-likelihood at this lambda, then exit.
    #[arg(long, value_name = "L", allow_hyphen_values = true)]
    pub llf_at: Option<f64>,

    /// Objective for the optimal-lambda search when `--lambda` is absent.
    #[arg(long, value_name = "METHOD", default_value = "mle")]
    pub normmax_method: String,

    #[command(flatten)]
    pub common: rsomics_common::CommonFlags,
}

#[derive(Serialize)]
#[serde(untagged)]
enum Output {
    Llf { lambda: f64, llf: f64 },
    Transform { lambda: f64, transformed: Vec<f64> },
}

impl Cli {
    pub fn run(self) -> ExitCode {
        let common = self.common.clone();
        run(&common, META, || {
            let data = parse_values(&self.data)?;

            if let Some(lmb) = self.llf_at {
                let llf = boxcox_llf(lmb, &data);
                if !common.json {
                    println!("# lambda\t{lmb}");
                    println!("{llf}");
                }
                return Ok(Output::Llf { lambda: lmb, llf });
            }

            let lambda = match self.lambda {
                Some(l) => {
                    if data.iter().any(|&v| v <= 0.0) {
                        return Err(RsomicsError::InvalidInput("data must be positive".into()));
                    }
                    l
                }
                None => {
                    check_positive_nonconstant(&data)?;
                    let method = NormmaxMethod::parse(&self.normmax_method)?;
                    boxcox_normmax(&data, method)?
                }
            };

            let transformed = boxcox(&data, lambda);
            if !common.json {
                write_transformed(lambda, &transformed)?;
            }
            Ok(Output::Transform {
                lambda,
                transformed,
            })
        })
    }
}

/// Stream the transformed values through ryu into a single buffered writer; the
/// per-value path never allocates a `String`.
fn write_transformed(lambda: f64, values: &[f64]) -> rsomics_common::Result<()> {
    let stdout = std::io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    let mut buf = ryu::Buffer::new();
    writeln!(out, "# lambda\t{lambda}").map_err(RsomicsError::Io)?;
    for &v in values {
        out.write_all(buf.format(v).as_bytes())
            .map_err(RsomicsError::Io)?;
        out.write_all(b"\n").map_err(RsomicsError::Io)?;
    }
    out.flush().map_err(RsomicsError::Io)
}

#[cfg(test)]
mod tests {
    use clap::CommandFactory;

    #[test]
    fn cli_definition_is_valid() {
        super::Cli::command().debug_assert();
    }
}
