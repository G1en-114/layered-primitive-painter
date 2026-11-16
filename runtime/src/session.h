// ONNX Runtime sessions for the exported graphs, called by graph and tensor names.
#pragma once
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"

namespace lpp {

// Host tensor; dtype is ONNX_TENSOR_ELEMENT_DATA_TYPE_{FLOAT,INT64,BOOL}.
struct Tensor {
  ONNXTensorElementDataType dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  std::vector<int64_t> shape;
  std::vector<float> f;
  std::vector<int64_t> i;
  std::vector<uint8_t> b;                       // bool as one byte (ORT layout)
  static Tensor F(std::vector<int64_t> shape, float fill = 0.f);
  static Tensor I(std::vector<int64_t> shape, int64_t fill = 0);
  static Tensor B(std::vector<int64_t> shape, bool fill = false);
  int64_t numel() const;
};

class Graph {
 public:
  Graph(Ort::Env& env, const std::string& path, const Ort::SessionOptions& so, std::vector<std::string> inputs,
        std::vector<std::string> outputs);
  // inputs in manifest order; returns outputs in manifest order (float tensors)
  std::vector<Tensor> run(const std::vector<const Tensor*>& inputs);

 private:
  Ort::Session session_;
  std::vector<std::string> in_names_, out_names_;
  Ort::MemoryInfo mem_;
};

class Graphs {
 public:
  Graphs(const std::string& export_dir, const std::string& provider);
  Graph& operator[](const std::string& name) { return *graphs_.at(name); }

 private:
  Ort::Env env_;
  std::map<std::string, std::unique_ptr<Graph>> graphs_;
};

}  // namespace lpp
