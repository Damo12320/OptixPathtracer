#pragma once

#include"OptiX/OptixRenderer.h"
#include"OpenGLWindow.h"
#include"OptixViewTexture.h"

#include <memory>

struct OptixViewDefinition {
	std::string ptxPath;
};

class OptixView {
public:
	std::unique_ptr<OpenGLWindow> window;
	std::unique_ptr<OptixRenderer> optixRenderer;
	std::unique_ptr<Camera> camera;

	glm::ivec2 viewSize;

	bool isCameraMoving = false;
private:
	std::unique_ptr<OptixViewTexture> viewTexture;
public:
	OptixView(OptixViewDefinition viewDef, glm::ivec2 viewSize, Model* model, std::unique_ptr<Camera> camera);

	void Run();
	void Resize(int width, int height);
	void Resize(glm::ivec2 viewSize);

private:
	void GLFrameSetup();
	void DrawOptix();
	void DrawToWindow();
	void EndFrame();
};