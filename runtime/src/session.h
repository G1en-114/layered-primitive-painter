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

// Per-graph wall time and call counts, printed at exit when LPP_PROFILE=1.
struct Profile {
  static std::map<std::string, std::pair<double, long>>& table();
  static void add(const std::string& name, double seconds);
  static void report();
};

class Graph {
 public:
  Graph(Ort::Env& env, const std::string& path, const Ort::SessionOptions& so, std::vector<std::string> inputs,
        std::vector<std::string> outputs, std::string name = "");
  const std::string& name() const { return name_; }
  // inputs in manifest order; returns outputs in manifest order (float tensors)
  std::vector<Tensor> run(const std::vector<const Tensor*>& inputs);
  Ort::Session& session() { return session_; }
  const std::vector<std::string>& input_names() const { return in_names_; }
  const std::vector<std::string>& output_names() const { return out_names_; }
  // wrap a host tensor as an Ort::Value (no copy; the Tensor must outlive the value)
  Ort::Value value(const Tensor& t);

 private:
  Ort::Session session_;
  std::vector<std::string> in_names_, out_names_;
  Ort::MemoryInfo mem_;
  std::string name_;
};

class Graphs {
 public:
  Graphs(const std::string& export_dir, const std::string& provider);
  Graph& operator[](const std::string& name) { return *graphs_.at(name); }
  bool has(const std::string& name) const { return graphs_.count(name) > 0; }
  const std::string& provider() const { return provider_; }

 private:
  Ort::Env env_;
  std::string provider_;
  std::map<std::string, std::unique_ptr<Graph>> graphs_;
};

}  // namespace lpp
