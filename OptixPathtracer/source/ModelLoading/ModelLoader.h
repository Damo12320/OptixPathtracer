#pragma once
#include <string>
#include <iostream>
#include "Model.h"
#include "../3rdParty/tiny_gltf.h"

class ModelLoader {
public:
	static Model* LoadModel(std::string filePath);

private:
	static void ParseScene(Model* finalModel, tinygltf::Model& model, tinygltf::Scene& scene);
	static void ParseNodes(Model* finalModel, tinygltf::Model& model, tinygltf::Node& node);
	static void ParseTransformation(Mesh* finalMesh, tinygltf::Node& node);
};