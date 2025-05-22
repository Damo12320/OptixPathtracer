#pragma once
#include "Mesh.h";
#include "Texture.h"

class Model {
public:
	std::vector<std::unique_ptr<Mesh>> meshes;
	std::vector<std::unique_ptr<Texture>> textures;
};