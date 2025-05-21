#include"Renderer/OptixView.h"
#include"ModelLoading/ModelLoader.h"

int main()
{
    std::string modelPath{ "assets/Models/3/untitled.gltf" };
    //std::string modelPath{ "assets/Models/Cube/untitled.gltf" };
    Model* model = ModelLoader::LoadModel(modelPath);

    OptixViewDefinition viewDef;
    viewDef.ptxPath = "source/Renderer/OptiX/devicePrograms.cu.ptx";

    glm::ivec2 viewSize{ 1920, 1080 };
    OptixView* optixView = new OptixView(viewDef, viewSize, model);

    Camera camera;
    camera.at = glm::vec3(0, -1, 0);
    camera.from = glm::vec3(5, -1, 0);
    camera.up = glm::vec3(0, -1, 0);

    optixView->optixRenderer->SetCamera(camera);

    optixView->Run();
}
