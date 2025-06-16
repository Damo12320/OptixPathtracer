#include "Camera.h"

#include "../3rdParty/glm/gtc/matrix_transform.hpp"
#include "../GlmHelperMethods.h"

Camera::Camera() {
	this->horizontalFOV_radians = glm::radians(40.0f);
}


#pragma region Setter

void Camera::SetHorizontalFOV(float degrees) {
	this->horizontalFOV_radians = glm::radians(degrees);
}

void Camera::SetBlenderPosition(glm::vec3 blenderPosition) {
	//this->position = glm::vec3(blenderPosition.x, blenderPosition.z, -blenderPosition.y);
	this->position = GlmHelper::BlenderToEnginePosition(blenderPosition);
}

void Camera::SetBlenderRotation(glm::vec3 blenderRotation) {
	//this->rotation = glm::vec3(90 - blenderRotation.x, 180 + blenderRotation.z, blenderRotation.y);
	this->rotation = GlmHelper::BlenderToEngineRotation(blenderRotation);
}

#pragma endregion


#pragma region Getter


float Camera::GetHorizontalFOVRadians() {
	return this->horizontalFOV_radians;
}

glm::vec3 Camera::GetForward() {
	glm::vec3 rotationRadians = glm::radians(this->rotation);

	float x = std::sin(rotationRadians.y);
	x *= std::cos(rotationRadians.x);

	float y = -std::sin(rotationRadians.x);

	float z = std::cos(rotationRadians.x);
	z *= std::cos(rotationRadians.y);

	return glm::normalize(glm::vec3(x, y, z));
}

glm::vec3 Camera::GetRight() {
	glm::vec3 forward = this->GetForward();

	return glm::normalize(glm::cross(forward, this->worldUp));
}

glm::vec3 Camera::GetUp() {
	glm::vec3 forward = GetForward();
	glm::vec3 right = GetRight();

	return glm::normalize(glm::cross(right, forward));
}

glm::mat4x4 Camera::GetViewMatrix() {
	return glm::lookAt(this->position, this->position + this->GetForward(), this->worldUp);
}

glm::mat4x4 Camera::GetProjectionMatrix(float aspectRatio) {
	return glm::perspective<float>(this->horizontalFOV_radians, aspectRatio, 0.1, 100.0);
}

#pragma endregion