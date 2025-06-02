#pragma once

#include"OptiX/OptixRenderer.h"
#include"OpenGLWindow.h"
#include"OpenGL/GLTexture2D.h"
#include"OpenGL/Framebuffer.h"
#include"OpenGL/ShaderProgramm.h"

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

	bool clearFrameBuffer = false;
private:
	std::unique_ptr<Framebuffer> framebuffer;
	std::unique_ptr<GLTexture2D> newFrame;

	std::unique_ptr<ShaderProgramm> combineShader;
	std::unique_ptr<ShaderProgramm> finalShader;

	int samples = 0;
public:
	OptixView(OptixViewDefinition viewDef, glm::ivec2 viewSize, Model* model, std::unique_ptr<Camera> camera);

	void Run();
	void Resize(int width, int height);
	void Resize(glm::ivec2 viewSize);

private:
	void GLFrameSetup();
	void DrawOptix(GLTexture2D* texture);
	void AddNewFrameToBuffer(GLTexture2D* newFrame, Framebuffer* buffer);
	void DrawToWindow();
	void EndFrame();
};