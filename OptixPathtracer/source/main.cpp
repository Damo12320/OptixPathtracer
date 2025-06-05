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


    std::unique_ptr<Camera> camera = std::unique_ptr<Camera>(new Camera());
    //camera.position = glm::vec3(-0.977644f, 1.0745f, 0.366231f);
    //camera.rotation = glm::vec3(90 - 89.1897, 180 + 77.765, 0);
    camera->SetBlenderPosition(glm::vec3(-0.977644, -0.366231, 1.0745));
    camera->SetBlenderRotation(glm::vec3(89.1897, 20, 77.765));

    glm::ivec2 viewSize{ 1920, 1080 };
    //glm::ivec2 viewSize{ 1920 / 2, 1080 / 2 };
    OptixView* optixView = new OptixView(viewDef, viewSize, model.get(), std::move(camera));

    optixView->Run();
}
