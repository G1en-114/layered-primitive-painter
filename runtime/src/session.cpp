#include "session.h"

#include <fstream>
#include <stdexcept>

#include "json.hpp"

namespace lpp {

Tensor Tensor::F(std::vector<int64_t> s, float fill) {
  Tensor t;
  t.dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  t.shape = std::move(s);
  t.f.assign(static_cast<size_t>(t.numel()), fill);
  return t;
}
Tensor Tensor::I(std::vector<int64_t> s, int64_t fill) {
  Tensor t;
  t.dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  t.shape = std::move(s);
  t.i.assign(static_cast<size_t>(t.numel()), fill);
  return t;
}
Tensor Tensor::B(std::vector<int64_t> s, bool fill) {
  Tensor t;
  t.dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
  t.shape = std::move(s);
  t.b.assign(static_cast<size_t>(t.numel()), fill ? 1 : 0);
  return t;
}
int64_t Tensor::numel() const {
  int64_t n = 1;
  for (auto d : shape) n *= d;
  return n;
}

Graph::Graph(Ort::Env& env, const std::string& path, const Ort::SessionOptions& so, std::vector<std::string> inputs,
             std::vector<std::string> outputs)
    : session_(env, path.c_str(), so), in_names_(std::move(inputs)), out_names_(std::move(outputs)),
      mem_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {}

std::vector<Tensor> Graph::run(const std::vector<const Tensor*>& inputs) {
  if (inputs.size() != in_names_.size()) throw std::runtime_error("graph input count mismatch");
  std::vector<Ort::Value> vals;
  std::vector<const char*> in_c, out_c;
  for (size_t k = 0; k < inputs.size(); ++k) {
    const Tensor* t = inputs[k];
    void* data = nullptr;
    size_t bytes = 0;
    switch (t->dtype) {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: data = const_cast<float*>(t->f.data()); bytes = t->f.size() * 4; break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: data = const_cast<int64_t*>(t->i.data()); bytes = t->i.size() * 8; break;
      default: data = const_cast<uint8_t*>(t->b.data()); bytes = t->b.size(); break;
    }
    vals.push_back(Ort::Value::CreateTensor(mem_, data, bytes, t->shape.data(), t->shape.size(), t->dtype));
    in_c.push_back(in_names_[k].c_str());
  }
  for (auto& n : out_names_) out_c.push_back(n.c_str());
  auto outs = session_.Run(Ort::RunOptions{nullptr}, in_c.data(), vals.data(), vals.size(), out_c.data(), out_c.size());
  std::vector<Tensor> res;
  for (auto& o : outs) {
    auto info = o.GetTensorTypeAndShapeInfo();
    Tensor t = Tensor::F(info.GetShape());
    const float* p = o.GetTensorData<float>();
    std::copy(p, p + t.f.size(), t.f.begin());
    res.push_back(std::move(t));
  }
  return res;
}

Graphs::Graphs(const std::string& dir, const std::string& provider) : env_(ORT_LOGGING_LEVEL_WARNING, "lpp") {
  std::ifstream f(dir + "/manifest.json");
  if (!f) throw std::runtime_error("cannot open " + dir + "/manifest.json");
  const auto m = nlohmann::json::parse(f);
  Ort::SessionOptions so;
  so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  so.SetIntraOpNumThreads(4);
  if (provider == "cuda" || provider == "cuda-tf32") {
    // V2 options: ORT's CUDA EP runs FP32 matmuls in TF32 by default (10-bit mantissa), which
    // changes logits enough to fork sampled sequences within a few records. Parity runs keep
    // full FP32; "cuda-tf32" opts into the faster mode.
    const OrtApi& api = Ort::GetApi();
    OrtCUDAProviderOptionsV2* opt = nullptr;
    Ort::ThrowOnError(api.CreateCUDAProviderOptions(&opt));
    const char* keys[] = {"device_id", "use_tf32"};
    const char* vals[] = {"0", provider == "cuda-tf32" ? "1" : "0"};
    Ort::ThrowOnError(api.UpdateCUDAProviderOptions(opt, keys, vals, 2));
    so.AppendExecutionProvider_CUDA_V2(*opt);
    api.ReleaseCUDAProviderOptions(opt);
  } else if (provider != "cpu") {
    throw std::runtime_error("provider must be cpu, cuda or cuda-tf32");
  }
  for (auto& [name, g] : m["graphs"].items())
    graphs_[name] = std::make_unique<Graph>(env_, dir + "/" + g["file"].get<std::string>(), so,
                                            g["inputs"].get<std::vector<std::string>>(),
                                            g["outputs"].get<std::vector<std::string>>());
}

}  // namespace lpp
