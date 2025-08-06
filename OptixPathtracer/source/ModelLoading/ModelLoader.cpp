#include "ModelLoader.h"

 //Define these only in *one* .cc file.
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../3rdParty/tiny_gltf.h"

//public static
//public static
std::unique_ptr<Model> ModelLoader::LoadModel(std::string folderPath, std::string fileName) {
	std::cout << "Load Model at path: " << folderPath << fileName << std::endl;
	std::cout << "Only the first scene of the GLTF will be loaded!" << std::endl;

	tinygltf::Model model;
	tinygltf::TinyGLTF loader;
	std::string err;
	std::string warn;

	bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, folderPath + fileName);

	if (!warn.empty()) {
		std::cout << "Warn: " << warn.c_str() << std::endl;
	}

	if (!err.empty()) {
		std::cout << "Err: " << err.c_str() << std::endl;
	}

	if (!ret) {
		std::cout << "Failed to parse glTF" << std::endl;
	}

	std::unique_ptr<Model> finalModel(new Model());

	ParseScene(finalModel.get(), model, model.scenes[0]);

	LoadTextures(finalModel.get(), model, folderPath);


	std::cout << "Model loaded! \n" << std::endl;
	return std::move(finalModel);
}

//private static
void ModelLoader::LoadTextures(Model* finalModel, tinygltf::Model& model, std::string& folderPath) {
	std::cout << "Loading " << model.textures.size() << " Textures..." << std::endl;

	int textureCounter = 0;

	for (int i = 0; i < model.textures.size(); i++) {
		tinygltf::Texture texture = model.textures[i];

		std::string texturePath = model.images[texture.source].uri;

		texturePath = folderPath + texturePath;

		//std::cout << "TexturePath " << texturePath << std::endl;

		glm::ivec2 res;
		int channels;
		float* image = stbi_loadf(texturePath.c_str(), &res.x, &res.y, &channels, STBI_rgb_alpha);
		
		if (image) {
			std::unique_ptr<Texture> texture(new Texture);
			texture->resolution = res;

			texture->pixel = image;

			//this is said by the Optix Sample. I couldn't see the issue
			/* iw - actually, it seems that stbi loads the pictures mirrored along the y axis - mirror them here */
			/*for (int y = 0; y < res.y / 2; y++) {
				uint32_t* line_y = texture->pixel + y * res.x;
				uint32_t* mirrored_y = texture->pixel + (res.y - 1 - y) * res.x;
				int mirror_y = res.y - 1 - y;
				for (int x = 0; x < res.x; x++) {
					std::swap(line_y[x], mirrored_y[x]);
				}
			}*/

			textureCounter++;
			finalModel->textures.push_back(std::move(texture));
		}
	}

	std::cout << "Loaded " << textureCounter << " Textures" << std::endl;
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
		//Every Primitive is its own Mesh (only has one Material per Mesh)
		std::unique_ptr<Mesh> finalMesh(new Mesh());

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
		if (primitive.attributes.find("TEXCOORD_0") != primitive.attributes.end()) {
			accessor = model.accessors[primitive.attributes["TEXCOORD_0"]];
			bufferView = model.bufferViews[accessor.bufferView];
			buffer = model.buffers[bufferView.buffer];
			const float* texCoords = reinterpret_cast<const float*>(&buffer.data[bufferView.byteOffset + accessor.byteOffset]);

			for (size_t i = 0; i < accessor.count; ++i) {
				glm::vec2 texCoord{ texCoords[i * 2 + 0] , texCoords[i * 2 + 1] };
				finalMesh->texCoord.push_back(texCoord);
			}
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

		ParseMaterial(finalMesh.get(), model, primitive);
		ParseTransformation(finalMesh.get(), node);

		finalMesh->CalculateTangentBasis();

		finalMesh->meshName = node.name;

		finalModel->meshes.push_back(std::move(finalMesh));
	}

	//for (auto mesh : finalModel->meshes)
		//for (auto vtx : mesh->vertecies)
			//finalModel->bounds.extend(vtx);
}

void ModelLoader::ParseMaterial(Mesh* finalMesh, tinygltf::Model& model, tinygltf::Primitive& primitive) {
	//Does this mesh has a material?
	if (model.materials.size() <= 0 || primitive.material < 0) {
		finalMesh->albedo = glm::vec3(1, 1, 1);
		return;
	}

	tinygltf::Material& material = model.materials[primitive.material];

	//Textures
	finalMesh->albedoTex = material.pbrMetallicRoughness.baseColorTexture.index;
	finalMesh->metalRoughTex = material.pbrMetallicRoughness.metallicRoughnessTexture.index;
	finalMesh->normalTex = material.normalTexture.index;

	//Material properties
	finalMesh->metallic = material.pbrMetallicRoughness.metallicFactor;
	finalMesh->roughness = material.pbrMetallicRoughness.roughnessFactor;

	glm::vec3 baseColor{ static_cast<float>(material.pbrMetallicRoughness.baseColorFactor[0])
						, static_cast<float>(material.pbrMetallicRoughness.baseColorFactor[1])
						, static_cast<float>(material.pbrMetallicRoughness.baseColorFactor[2]) };

	finalMesh->albedo = baseColor;

	//Random Color for each Mesh
	//finalMesh->albedo = glm::vec3(((float)std::rand()) / RAND_MAX, ((float)std::rand()) / RAND_MAX, ((float)std::rand()) / RAND_MAX);
}

void ModelLoader::ParseTransformation(Mesh* finalMesh, tinygltf::Node& node) {
	//transation
	if (node.translation.size() == 0) {
		finalMesh->translation = glm::vec3(0);
	}
	else {
		finalMesh->translation = glm::vec3(
			node.translation[0], 
			node.translation[1], 
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
		/*finalMesh->rotation = glm::quat(
			node.rotation[0],
			node.rotation[1],
			node.rotation[2],
			node.rotation[3]
		);*/
		finalMesh->rotation = glm::quat(
			node.rotation[3],
			node.rotation[0],
			node.rotation[1],
			node.rotation[2]
		);
	}

	finalMesh->CalculateModelMatrix();
}