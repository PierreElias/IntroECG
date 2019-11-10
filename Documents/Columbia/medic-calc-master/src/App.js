import React, { Component } from "react";
import styled from "@emotion/styled";
import swap from "./swap.svg";

const StyledApp = styled.div`
  width: 40rem;
  margin: 5rem auto;

  h2 {
    text-align: center;
  }

  .result {
    margin: 2rem 0;
    font-size: 1.2rem;
    display: flex;
    justify-content: center;

    .result-head {
      background: green;
      padding: 2rem;
    }

    .result-body {
      padding: 2rem;
      background: #90ee90;
    }
  }
`;

const StyledForm = styled.form`
  display: flex;
  flex-direction: column;
  font-size: 1rem;

  label {
    padding-bottom: 1.5rem;
    padding-top: 1.5rem;
    display: flex;
    justify-content: space-between;
    border-top: 1px solid #d3d8df;

    .label-name {
      align-self: center;
    }
  }

  .last {
    border-bottom: 1px solid #d3d8df;
  }

  .input {
    display: flex;

    input {
      padding: 0.5rem 1rem;
      height: 2.5rem;
      width: 15rem;
      border-top-left-radius: 0.5rem;
      border-bottom-left-radius: 0.5rem;
      border: 1px solid #d3d8df;
      outline: none;
    }

    button {
      border-top-right-radius: 0.5rem;
      border-bottom-right-radius: 0.5rem;
      border: 1px solid #d3d8df;
      width: 8rem;
      outline: none;
      cursor: pointer;

      display: flex;
      justify-content: space-around;

      span {
        font-size: 1rem;
      }
      img {
        height: 1.5rem;
      }
    }
  }

  .submit {
    width: 20rem;
    align-self: center;
    margin-top: 2rem;
    height: 3rem;
    border: 1px solid #d3d8df;
    border-radius: 0.5rem;
    outline: none;
    cursor: pointer;
  }
`;

class App extends Component {
  state = {
    apButton: "g/L",
    tgButton: "mmol/L",
    tcButton: "mmol/L",
    apoB: "",
    tg: "",
    tc: "",
    result: ""
  };

  handleInputChange = e => {
    const { name, value } = e.target;
    this.setState({
      [name]: value
    });
  };

  handleSwitch = (event, type) => {
    event.preventDefault();
    const { apButton, tcButton, tgButton } = this.state;
    if (type === "ap" && apButton === "g/L") {
      return this.setState({
        apButton: "mg/dL"
      });
    } else if (type === "ap" && apButton === "mg/dL") {
      return this.setState({
        apButton: "g/L"
      });
    } else if (type === "tg" && tgButton === "mmol/L") {
      this.setState({
        tgButton: "mg/dL"
      });
    } else if (type === "tg" && tgButton === "mg/dL") {
      this.setState({
        tgButton: "mmol/L"
      });
    } else if (type === "tc" && tcButton === "mmol/L") {
      this.setState({
        tcButton: "mg/dL"
      });
    } else if (type === "tc" && tcButton === "mg/dL") {
      this.setState({
        tcButton: "mmol/L"
      });
    }
  };

  apoBConvertToGL = () => {
    const { apoB, apButton } = this.state;
    if (apButton === "g/L") {
      return apoB;
    } else {
      return this.convertMGDLToGL(apoB);
    }
  };

  tgConvertToMMOL = () => {
    const { tg, tgButton } = this.state;
    if (tgButton === "mmol/L") {
      return tg;
    } else {
      return this.convertMMOLToMGDL(tg);
    }
  };

  tcConvertToMMOL = () => {
    const { tc, tcButton } = this.state;
    if (tcButton === "mmol/L") {
      return tc;
    } else {
      return this.convertMMOLToMGDL(tc);
    }
  };

  calculate = () => {
    const { apoB, tc, tg } = this.state;

    //Normal Apo B < 1.2g/l
    if (this.apoBConvertToGL(apoB) < 1.2) {
      if (this.tgConvertToMMOL(tg) < 1.5) {
        return "Normal";
      } else {
        // Hyper TG>= 1.5mmol
        if (this.tgConvertToMMOL(tg) >= 10) {
          // TG: ABpoB
          if (this.apoBConvertToGL(apoB) >= 0.75) {
            return "Chylomicrons + VLDL, Type 1 hyperchylomicronemia";
          } else {
            return "Chylomicrons, Type 1, Type 1 hyperchylomicronemia";
          }
        } else {
          // TG:Apob < 10
          if (this.tcConvertToMMOL(tc) >= 0.2) {
            return "Chylomicrons + VLDL remnants, Type 3 Familial dysbetalipoproteinemia";
          } else {
            return "VLDL, type V Familial hypertriglyceridemia";
          }
        }
      }
      //Hyper Apo B >= 1.2g/l
    } else {
      // Normal TG
      if (this.tgConvertToMMOL(tg) < 1.5) {
        return "LDL, type IIa Familial/polygenic hypercholesterolemia";
      } else {
        // Hyper TG
        return "VLDL+LDL, type IIb combined hyperlipidemia";
      }
    }
  };

  convertMGDLToGL = number => {
    return number / 100;
  };

  convertGLToMGDL = number => {
    return number * 100;
  };

  convertMGDLToMMOL = number => {
    return number * 18.018018;
  };

  convertMMOLToMGDL = number => {
    return number / 18.018018;
  };

  render() {
    const { apButton, tcButton, tgButton, apoB, tc, tg, result } = this.state;
    console.log(this.apoBConvertToGL(apoB));
    return (
      <StyledApp>
        <h2>Calculator</h2>

        <StyledForm>
          <label htmlFor="">
            <span className="label-name">Apolipoprotein B</span>
            <div className="input">
              <input
                required
                type="number"
                name="apoB"
                value={apoB}
                onChange={e => this.handleInputChange(e)}
                tabindex="1"
                autoFocus
              />
              <button
                onClick={e => {
                  this.handleSwitch(e, "ap");

                  if (apButton === "g/L") {
                    this.setState({
                      apoB: this.convertGLToMGDL(apoB)
                    });
                  } else {
                    this.setState({
                      apoB: this.convertMGDLToGL(apoB)
                    });
                  }
                }}
              >
                <span>{apButton}</span> <img src={swap} alt="" />
              </button>
            </div>
          </label>

          <label htmlFor="" className="last">
            <span className="label-name">Trigylcerides (Tri)</span>
            <div className="input">
              <input
                required
                type="number"
                name="tg"
                value={tg}
                tabindex="2"
                onChange={e => this.handleInputChange(e)}
              />
              <button
                onClick={e => {
                  this.handleSwitch(e, "tg");
                  if (tgButton === "mmol/L") {
                    this.setState({
                      tg: this.convertMGDLToMMOL(tg)
                    });
                  } else {
                    this.setState({
                      tg: this.convertMMOLToMGDL(tg)
                    });
                  }
                }}
              >
                <span>{tgButton}</span> <img src={swap} alt="" />
              </button>
            </div>
          </label>

          <label htmlFor="">
            <span className="label-name">Total Cholesterol (TChol)</span>
            <div className="input">
              <input
                required
                type="number"
                name="tc"
                value={tc}
                tabindex="3"
                onChange={e => this.handleInputChange(e)}
              />
              <button
                onClick={e => {
                  this.handleSwitch(e, "tc");
                  if (tcButton === "mmol/L") {
                    this.setState({
                      tc: this.convertMGDLToMMOL(tc)
                    });
                  } else {
                    this.setState({
                      tc: this.convertMMOLToMGDL(tc)
                    });
                  }
                }}
              >
                <span>{tcButton}</span> <img src={swap} alt="" />
              </button>
            </div>
          </label>
          <button
            className="submit"
            onClick={e => {
              if (tg && apoB && tc) {
                e.preventDefault();

                this.setState({
                  result: this.calculate()
                });
              }
            }}
          >
            Calculate
          </button>
        </StyledForm>
        {result && (
          <div className="result">
            <span className="result-head">Result:</span>
            <span className="result-body">{result}</span>
          </div>
        )}
      </StyledApp>
    );
  }
}

export default App;
