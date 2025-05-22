#include"Renderer/OptixView.h"
#include"ModelLoading/ModelLoader.h"

int main()
{
    //std::string modelPath{ "assets/Models/3/untitled.gltf" };
    std::string modelPath{ "assets/Models/Sponza/" };
    //std::string modelPath{ "assets/Models/Cube + Sphere/untitled.gltf" };
    //std::string modelPath{ "assets/Models/Cube + Sphere Yup/untitled.gltf" };
    //std::string modelPath{ "assets/Models/Cube/untitled.gltf" };
    std::unique_ptr<Model> model = ModelLoader::LoadModel(modelPath, "untitled.gltf");

    OptixViewDefinition viewDef;
    viewDef.ptxPath = "source/Renderer/OptiX/devicePrograms.cu.ptx";

    glm::ivec2 viewSize{ 1920, 1080 };
    OptixView* optixView = new OptixView(viewDef, viewSize, model.get());

    Camera camera;
    camera.at = glm::vec3(0, 2, 0);
    camera.from = glm::vec3(5, 2, 0);
    camera.up = glm::vec3(0, 1, 0);

    optixView->optixRenderer->SetCamera(camera);

    optixView->Run();
}
