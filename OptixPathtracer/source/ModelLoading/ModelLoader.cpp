#include "ModelLoader.h"

 //Define these only in *one* .cc file.
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../3rdParty/tiny_gltf.h"

//public static
//public static
Model* ModelLoader::LoadModel(std::string filePath) {
	std::cout << "Load Model at path: " << filePath << std::endl;

	tinygltf::Model model;
	tinygltf::TinyGLTF loader;
	std::string err;
	std::string warn;

	bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, filePath);

	if (!warn.empty()) {
		std::cout << "Warn: " << warn.c_str() << std::endl;
	}

	if (!err.empty()) {
		std::cout << "Err: " << err.c_str() << std::endl;
	}

	if (!ret) {
		std::cout << "Failed to parse glTF" << std::endl;
	}

	Model* finalModel = new Model();
	ParseScene(finalModel, model, model.scenes[0]);

	return finalModel;
}

//private static
void ModelLoader::ParseScene(Model* finalModel, tinygltf::Model& model, tinygltf::Scene& scene) {
	for (int nodeID : scene.nodes) {
		ParseNodes(finalModel, model, model.nodes[nodeID]);
	}
}

//private static
void ModelLoader::ParseNodes(Model* finalModel, tinygltf::Model& model, tinygltf::Node& node) {
	tinygltf::Mesh& mesh = model.meshes[node.mesh];

	for (tinygltf::Primitive& primitive : mesh.primitives) {
		Mesh* finalMesh = new Mesh();

		//Positions
		tinygltf::Accessor accessor = model.accessors[primitive.attributes["POSITION"]];
		tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
		tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];
		const float* positions = reinterpret_cast<const float*>(&buffer.data[bufferView.byteOffset + accessor.byteOffset]);

		for (size_t i = 0; i < accessor.count; ++i) {
			glm::vec3 vertex{ positions[i * 3 + 0] , positions[i * 3 + 1] , positions[i * 3 + 2] };
			finalMesh->vertecies.push_back(vertex);
		}


		//Normals
		accessor = model.accessors[primitive.attributes["NORMAL"]];
		bufferView = model.bufferViews[accessor.bufferView];
		buffer = model.buffers[bufferView.buffer];
		const float* normals = reinterpret_cast<const float*>(&buffer.data[bufferView.byteOffset + accessor.byteOffset]);

		for (size_t i = 0; i < accessor.count; ++i) {
			glm::vec3 normal{ normals[i * 3 + 0] , normals[i * 3 + 1] , normals[i * 3 + 2] };
			finalMesh->normal.push_back(normal);
		}


		//TexCoords
		accessor = model.accessors[primitive.attributes["TEXCOORD_0"]];
		bufferView = model.bufferViews[accessor.bufferView];
		buffer = model.buffers[bufferView.buffer];
		const float* texCoords = reinterpret_cast<const float*>(&buffer.data[bufferView.byteOffset + accessor.byteOffset]);

		for (size_t i = 0; i < accessor.count; ++i) {
			glm::vec2 texCoord{ texCoords[i * 2 + 0] , texCoords[i * 2 + 1] };
			finalMesh->texCoord.push_back(texCoord);
		}

		//Indicies
		accessor = model.accessors[primitive.indices];
		bufferView = model.bufferViews[accessor.bufferView];
		buffer = model.buffers[bufferView.buffer];
		//const int* indicies = reinterpret_cast<const int*>(&buffer.data[bufferView.byteOffset + accessor.byteOffset]);
		uint16_t* indices = reinterpret_cast<uint16_t*>(&buffer.data[accessor.byteOffset + bufferView.byteOffset]);

		//componenttype is TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT

		for (size_t i = 0; i < accessor.count; i += 3) {
			glm::ivec3 index{ indices[i + 0]
							, indices[i + 1]
							, indices[i + 2] };
			finalMesh->index.push_back(index);
		}

		if (model.materials.size() > 0) {
			//Material
			tinygltf::Material& material = model.materials[primitive.material];

			glm::vec3 baseColor{ static_cast<float>(material.pbrMetallicRoughness.baseColorFactor[0])
								, static_cast<float>(material.pbrMetallicRoughness.baseColorFactor[1])
								, static_cast<float>(material.pbrMetallicRoughness.baseColorFactor[2]) };

			finalMesh->albedo = baseColor;
		}
		else {
			finalMesh->albedo = glm::vec3(1, 1, 1);
		}
		ParseTransformation(finalMesh, node);

		finalMesh->meshName = node.name;

		finalModel->meshes.push_back(finalMesh);
	}

	//for (auto mesh : finalModel->meshes)
		//for (auto vtx : mesh->vertecies)
			//finalModel->bounds.extend(vtx);
}

void ModelLoader::ParseTransformation(Mesh* finalMesh, tinygltf::Node& node) {
	//transation
	if (node.translation.size() == 0) {
		finalMesh->translation = glm::vec3(0);
	}
	else {
		finalMesh->translation = glm::vec3(
			node.translation[0], 
			-node.translation[1], 
			node.translation[2]
		);
	}

	//scale
	if (node.scale.size() == 0) {
		finalMesh->scale = glm::vec3(1, 1, 1);
	}
	else {

		finalMesh->scale = glm::vec3(
			node.scale[0], 
			node.scale[1],
			node.scale[2]
		);
	}

	//rotaion
	if (node.rotation.size() == 0) {
		finalMesh->rotation = glm::quat();
	}
	else {
		finalMesh->rotation = glm::quat(
			node.rotation[0],
			node.rotation[1],
			node.rotation[2],
			node.rotation[3]
		);
	}

	finalMesh->CalculateModelMatrix();
}