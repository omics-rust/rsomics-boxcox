//! `scipy.optimize.brent` plus its downhill `bracket`, ported faithfully.
//!
//! `boxcox`/`boxcox_normmax` find the optimal lambda by minimizing a scalar
//! objective with `optimize.brent(func, brack=(-2.0, 2.0))`. The lambda scipy
//! reports is the path taken by Brent's method, so to land on the same value
//! within an optimizer's tolerance the bracketing search and the parabolic /
//! golden-section iteration must follow the same branches in the same order.

const GOLD: f64 = 1.618_034;
const VERYSMALL: f64 = 1e-21;
const GROW_LIMIT: f64 = 110.0;

const BRENT_TOL: f64 = 1.48e-8;
const MINTOL: f64 = 1.0e-11;
const CG: f64 = 0.381_966_0;
const MAXITER: usize = 500;

struct Bracket {
    xa: f64,
    xb: f64,
    xc: f64,
    fb: f64,
}

/// Downhill bracket search (`scipy.optimize.bracket`) seeded with `xa`, `xb`.
fn bracket(f: &mut impl FnMut(f64) -> f64, xa0: f64, xb0: f64) -> Bracket {
    let (mut xa, mut xb) = (xa0, xb0);
    let mut fa = f(xa);
    let mut fb = f(xb);
    if fa < fb {
        std::mem::swap(&mut xa, &mut xb);
        std::mem::swap(&mut fa, &mut fb);
    }
    let mut xc = xb + GOLD * (xb - xa);
    let mut fc = f(xc);

    let mut iter = 0usize;
    while fc < fb {
        let tmp1 = (xb - xa) * (fb - fc);
        let tmp2 = (xb - xc) * (fb - fa);
        let val = tmp2 - tmp1;
        let denom = if val.abs() < VERYSMALL {
            2.0 * VERYSMALL
        } else {
            2.0 * val
        };
        let mut w = xb - ((xb - xc) * tmp2 - (xb - xa) * tmp1) / denom;
        let wlim = xb + GROW_LIMIT * (xc - xb);

        if iter > 1000 {
            panic!("bracket: iteration limit reached");
        }
        iter += 1;

        if (w - xc) * (xb - w) > 0.0 {
            let fw = f(w);
            if fw < fc {
                xa = xb;
                xb = w;
                fb = fw;
                break;
            } else if fw > fb {
                xc = w;
                break;
            }
            w = xc + GOLD * (xc - xb);
            let fw = f(w);
            shift(&mut xa, &mut xb, &mut xc, w, &mut fa, &mut fb, &mut fc, fw);
        } else if (w - wlim) * (wlim - xc) >= 0.0 {
            w = wlim;
            let fw = f(w);
            shift(&mut xa, &mut xb, &mut xc, w, &mut fa, &mut fb, &mut fc, fw);
        } else if (w - wlim) * (xc - w) > 0.0 {
            let fw = f(w);
            if fw < fc {
                xb = xc;
                xc = w;
                let wn = xc + GOLD * (xc - xb);
                fb = fc;
                fc = fw;
                let fwn = f(wn);
                shift(
                    &mut xa, &mut xb, &mut xc, wn, &mut fa, &mut fb, &mut fc, fwn,
                );
            } else {
                shift(&mut xa, &mut xb, &mut xc, w, &mut fa, &mut fb, &mut fc, fw);
            }
        } else {
            w = xc + GOLD * (xc - xb);
            let fw = f(w);
            shift(&mut xa, &mut xb, &mut xc, w, &mut fa, &mut fb, &mut fc, fw);
        }
    }

    Bracket { xa, xb, xc, fb }
}

#[allow(clippy::too_many_arguments)]
fn shift(
    xa: &mut f64,
    xb: &mut f64,
    xc: &mut f64,
    w: f64,
    fa: &mut f64,
    fb: &mut f64,
    fc: &mut f64,
    fw: f64,
) {
    *xa = *xb;
    *xb = *xc;
    *xc = w;
    *fa = *fb;
    *fb = *fc;
    *fc = fw;
}

/// Minimize `f` with Brent's method bracketed by `(xa0, xb0)`; returns the
/// minimizing argument (`OptimizeResult.x` in scipy).
pub fn brent(mut f: impl FnMut(f64) -> f64, xa0: f64, xb0: f64) -> f64 {
    let br = bracket(&mut f, xa0, xb0);

    let mut x = br.xb;
    let mut w = br.xb;
    let mut v = br.xb;
    let mut fx = br.fb;
    let mut fw = br.fb;
    let mut fv = br.fb;
    let (mut a, mut b) = if br.xa < br.xc {
        (br.xa, br.xc)
    } else {
        (br.xc, br.xa)
    };

    let mut deltax = 0.0_f64;
    let mut rat = 0.0_f64;
    let mut iter = 0usize;

    while iter < MAXITER {
        let tol1 = BRENT_TOL * x.abs() + MINTOL;
        let tol2 = 2.0 * tol1;
        let xmid = 0.5 * (a + b);
        if (x - xmid).abs() < (tol2 - 0.5 * (b - a)) {
            break;
        }

        if deltax.abs() <= tol1 {
            deltax = if x >= xmid { a - x } else { b - x };
            rat = CG * deltax;
        } else {
            let tmp1 = (x - w) * (fx - fv);
            let mut tmp2 = (x - v) * (fx - fw);
            let mut p = (x - v) * tmp2 - (x - w) * tmp1;
            tmp2 = 2.0 * (tmp2 - tmp1);
            if tmp2 > 0.0 {
                p = -p;
            }
            tmp2 = tmp2.abs();
            let dx_temp = deltax;
            deltax = rat;
            if p > tmp2 * (a - x) && p < tmp2 * (b - x) && p.abs() < (0.5 * tmp2 * dx_temp).abs() {
                rat = p / tmp2;
                let u = x + rat;
                if (u - a) < tol2 || (b - u) < tol2 {
                    rat = if xmid - x >= 0.0 { tol1 } else { -tol1 };
                }
            } else {
                deltax = if x >= xmid { a - x } else { b - x };
                rat = CG * deltax;
            }
        }

        let u = if rat.abs() < tol1 {
            if rat >= 0.0 { x + tol1 } else { x - tol1 }
        } else {
            x + rat
        };
        let fu = f(u);

        if fu > fx {
            if u < x {
                a = u;
            } else {
                b = u;
            }
            if fu <= fw || w == x {
                v = w;
                w = u;
                fv = fw;
                fw = fu;
            } else if fu <= fv || v == x || v == w {
                v = u;
                fv = fu;
            }
        } else {
            if u >= x {
                a = x;
            } else {
                b = x;
            }
            v = w;
            w = x;
            x = u;
            fv = fw;
            fw = fx;
            fx = fu;
        }

        iter += 1;
    }

    x
}
