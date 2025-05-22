#pragma once
#include <string>
#include <iostream>
#include "Model.h"
#include "../3rdParty/tiny_gltf.h"

class ModelLoader {
public:
	static std::unique_ptr<Model> LoadModel(std::string folderPath, std::string fileName);

private:
	static void LoadTextures(Model* finalModel, tinygltf::Model& model, std::string& folderPath);

	static void ParseScene(Model* finalModel, tinygltf::Model& model, tinygltf::Scene& scene);
	static void ParseNodes(Model* finalModel, tinygltf::Model& model, tinygltf::Node& node);

	static void ParseMaterial(Mesh* finalMesh, tinygltf::Model& model, tinygltf::Primitive& primitive);
	static void ParseTransformation(Mesh* finalMesh, tinygltf::Node& node);
};