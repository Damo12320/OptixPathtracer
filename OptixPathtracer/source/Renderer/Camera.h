#pragma once
#include "../3rdParty/glm/glm.hpp"

class Camera {
public:
    glm::vec3 worldUp{ 0, 1, 0 };

    glm::vec3 position{ 0, 0, 0 };
    glm::vec3 rotation{ 0, 0, 0 };

    float flySpeed = 1;

private:
    float horizontalFOV_radians;

public:
    Camera();

    void SetHorizontalFOV(float degrees);
    void SetBlenderPosition(glm::vec3 blenderPosition);
    void SetBlenderRotation(glm::vec3 blenderRotation);

    float GetHorizontalFOVRadians();
    glm::vec3 GetForward();
    glm::vec3 GetRight();
    glm::vec3 GetUp();

    glm::mat4x4 GetViewMatrix();
    glm::mat4x4 GetProjectionMatrix(float aspectRatio);
};