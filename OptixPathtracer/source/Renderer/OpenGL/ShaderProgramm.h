#pragma once

#include <glad/glad.h>

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

#include "../../3rdParty/glm/glm.hpp"
#include "../../3rdParty/glm/gtc/matrix_transform.hpp"
#include "../../3rdParty/glm/gtc/type_ptr.hpp"

class ShaderProgramm {
public:
    unsigned int ID;

public:
    ShaderProgramm(const char* vertexPath, const char* fragmentPath);

    void Bind();

    // utility uniform functions
    void SetBool(const std::string& name, bool value) const;
    void SetInt(const std::string& name, int value) const;
    void SetFloat(const std::string& name, float value) const;
    void SetTextureLocation(const std::string& name, GLuint textureLocation) const;
    void SetMatrix4x4(const std::string& name, glm::mat4 matrix) const;
    void SetVec3(const std::string& name, glm::vec3 value) const;

private:
    void CheckCompileErrors(unsigned int shader, std::string type);
};