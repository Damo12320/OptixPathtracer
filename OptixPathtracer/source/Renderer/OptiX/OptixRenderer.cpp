#include "OptixRenderer.h"

#include <iostream>
#include <fstream>
#include <optix_function_table_definition.h>

#include "MeshSBTData.h"

OptixRenderer::OptixRenderer(const std::string& ptxPath, Model* model) {
    this->model = model;

    this->InitOptix();
    this->CreateContext();

    //Read shader programs
    this->CreateModule(ptxPath);

    //Create shader programs
    this->CreateRaygenPrograms();
    this->CreateMissPrograms();
    this->CreateHitgroupPrograms();

    launchParams.traversable = this->BuildAccel();

    //create programs pipline
    this->CreatePipeline();

    this->CreateTextures();

    //build Shader Binding Table (which program does which raytype and what data does it have)
    this->BuildSBT();

    launchParamsBuffer.alloc(sizeof(launchParams));

    launchParams.camera.position = glm::vec3(1, 0, 0);
}

#pragma region OptiX Initialization

// helper function that initializes optix and checks for errors
void OptixRenderer::InitOptix() {
    std::cout << "initializing optix..." << std::endl;

    // -------------------------------------------------------
    // check for available optix7 capable devices
    // -------------------------------------------------------
    cudaFree(0);
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    if (numDevices == 0)
        throw std::runtime_error("no CUDA capable devices found!");
    std::cout << "found " << numDevices << " CUDA devices" << std::endl;

    // -------------------------------------------------------
    // initialize optix
    // -------------------------------------------------------
    OPTIX_CHECK(optixInit());
    std::cout << "successfully initialized optix... yay!" << std::endl;
}


// helper for context creation
static void context_log_cb(unsigned int level, const char* tag, const char* message, void*) {
    fprintf(stderr, "[%2d][%12s]: %s\n", (int)level, tag, message);
}

// creates and configures a optix device context (in this simple example, only for the primary GPU device)
void OptixRenderer::CreateContext() {
    // for this sample, do everything on one device
    const int deviceID = 0;
    CUDA_CHECK(SetDevice(deviceID));
    CUDA_CHECK(StreamCreate(&stream));

    cudaGetDeviceProperties(&deviceProps, deviceID);
    std::cout << "running on device: " << deviceProps.name << std::endl;

    CUresult  cuRes = cuCtxGetCurrent(&cudaContext);
    if (cuRes != CUDA_SUCCESS)
        fprintf(stderr, "Error querying current context: error code %d\n", cuRes);

    OptixDeviceContextOptions options = {};
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;

    OPTIX_CHECK(optixDeviceContextCreate(cudaContext, &options, &optixContext));
    OPTIX_CHECK(optixDeviceContextSetLogCallback(optixContext, context_log_cb, nullptr, 4));
}

// creates the module that contains all the programs we are going to use. 
// we use read a .ptx file that is created from the .cu file during the build process
void OptixRenderer::CreateModule(const std::string& ptxPath) {
    moduleCompileOptions.maxRegisterCount = 50;
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    pipelineCompileOptions = {};
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.numPayloadValues = 2;
    pipelineCompileOptions.numAttributeValues = 2;
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";

    pipelineLinkOptions.maxTraceDepth = 2;


    //read ptx file
    //------------------------------------------------------------------------------------------------------
    std::ifstream infile{ ptxPath };
    std::string cuCode{ std::istreambuf_iterator<char>(infile), std::istreambuf_iterator<char>() };
    infile.close();

    std::string ptxCode{ cuCode.c_str() };
    //------------------------------------------------------------------------------------------------------

    char log[2048];
    size_t sizeof_log = sizeof(log);

    OPTIX_CHECK(optixModuleCreateFromPTX(optixContext,
        &moduleCompileOptions,
        &pipelineCompileOptions,
        ptxCode.c_str(),
        ptxCode.size(),
        log,      // Log string
        &sizeof_log,// Log string sizse
        &module
    ));

    if (sizeof_log > 1) {
        std::cout << log << std::endl;
    }
}



// does all setup for the raygen program(s) we are going to use
void OptixRenderer::CreateRaygenPrograms() {
    // we do a single ray gen program in this example:
    raygenPGs.resize(1);

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc.raygen.module = module;
    pgDesc.raygen.entryFunctionName = "__raygen__renderFrame";

    // OptixProgramGroup raypg;
    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
        &pgDesc,
        1,
        &pgOptions,
        log, &sizeof_log,
        &raygenPGs[0]
    ));

    if (sizeof_log > 1) {
        std::cout << log << std::endl;
    }
}

// does all setup for the miss program(s) we are going to use
void OptixRenderer::CreateMissPrograms() {
    // we do a single ray gen program in this example:
    missPGs.resize(RAY_TYPE_COUNT);

    char log[2048];
    size_t sizeof_log = sizeof(log);

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgDesc.miss.module = module;

    // ------------------------------------------------------------------
    // radiance rays
    // ------------------------------------------------------------------
    pgDesc.miss.entryFunctionName = "__miss__radiance";

    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
        &pgDesc,
        1,
        &pgOptions,
        log, &sizeof_log,
        &missPGs[RADIANCE_RAY_TYPE]
    ));
    if (sizeof_log > 1) {
        std::cout << log << std::endl;
    }

    // ------------------------------------------------------------------
    // shadow rays
    // ------------------------------------------------------------------
    pgDesc.miss.entryFunctionName = "__miss__shadow";

    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
        &pgDesc,
        1,
        &pgOptions,
        log, &sizeof_log,
        &missPGs[SHADOW_RAY_TYPE]
    ));

    if (sizeof_log > 1) {
        std::cout << log << std::endl;
    }
}

// does all setup for the hitgroup program(s) we are going to use
void OptixRenderer::CreateHitgroupPrograms() {
    // for this simple example, we set up a single hit group
    hitgroupPGs.resize(RAY_TYPE_COUNT);

    char log[2048];
    size_t sizeof_log = sizeof(log);

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc    pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgDesc.hitgroup.moduleCH = module;
    pgDesc.hitgroup.moduleAH = module;

    // -------------------------------------------------------
    // radiance rays
    // -------------------------------------------------------
    pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";

    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
        &pgDesc,
        1,
        &pgOptions,
        log, &sizeof_log,
        &hitgroupPGs[RADIANCE_RAY_TYPE]
    ));
    if (sizeof_log > 1) {
        std::cout << log << std::endl;
    }

    // -------------------------------------------------------
    // shadow rays: technically we don't need this hit group,
    // since we just use the miss shader to check if we were not
    // in shadow
    // -------------------------------------------------------
    pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__shadow";
    pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__shadow";

    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
        &pgDesc,
        1,
        &pgOptions,
        log, &sizeof_log,
        &hitgroupPGs[SHADOW_RAY_TYPE]
    ));

    if (sizeof_log > 1) {
        std::cout << log << std::endl;
    }
}

// assembles the full pipeline of all programs
void OptixRenderer::CreatePipeline() {
    std::vector<OptixProgramGroup> programGroups;
    for (auto pg : raygenPGs)
        programGroups.push_back(pg);
    for (auto pg : missPGs)
        programGroups.push_back(pg);
    for (auto pg : hitgroupPGs)
        programGroups.push_back(pg);

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixPipelineCreate(optixContext,
        &pipelineCompileOptions,
        &pipelineLinkOptions,
        programGroups.data(),
        (int)programGroups.size(),
        log, &sizeof_log,
        &pipeline
    ));

    if (sizeof_log > 1) {
        std::cout << log << std::endl;
    }

    OPTIX_CHECK(optixPipelineSetStackSize
    (/* [in] The pipeline to configure the stack size for */
        pipeline,
        /* [in] The direct stack size requirement for direct
           callables invoked from IS or AH. */
        2 * 1024,
        /* [in] The direct stack size requirement for direct
           callables invoked from RG, MS, or CH.  */
        2 * 1024,
        /* [in] The continuation stack requirement. */
        2 * 1024,
        /* [in] The maximum depth of a traversable graph
           passed to trace. */
        1));

    if (sizeof_log > 1) {
        std::cout << log << std::endl;
    }
}

OptixTraversableHandle OptixRenderer::BuildAccel()
{
    const int numMeshes = (int)model->meshes.size();
    vertexBuffer.resize(numMeshes);
    normalBuffer.resize(numMeshes);
    texcoordBuffer.resize(numMeshes);
    indexBuffer.resize(numMeshes);
    matrices.resize(numMeshes);

    OptixTraversableHandle asHandle{ 0 };

    // ==================================================================
    // triangle inputs
    // ==================================================================
    std::vector<OptixBuildInput> triangleInput(numMeshes);
    std::vector<CUdeviceptr> d_vertices(numMeshes);
    std::vector<CUdeviceptr> d_indices(numMeshes);
    std::vector<uint32_t> triangleInputFlags(numMeshes);

    for (int meshID = 0; meshID < numMeshes; meshID++) {
        // upload the model to the device: the builder
        Mesh& mesh = *model->meshes[meshID];
        vertexBuffer[meshID].alloc_and_upload(mesh.vertecies);
        indexBuffer[meshID].alloc_and_upload(mesh.index);
        if (!mesh.normal.empty())
            normalBuffer[meshID].alloc_and_upload(mesh.normal);
        if (!mesh.texCoord.empty())
            texcoordBuffer[meshID].alloc_and_upload(mesh.texCoord);

        matrices[meshID].alloc_and_upload(mesh.ModelMatrix);

        triangleInput[meshID] = {};
        triangleInput[meshID].type
            = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

        // create local variables, because we need a *pointer* to the
        // device pointers
        d_vertices[meshID] = vertexBuffer[meshID].d_pointer();
        d_indices[meshID] = indexBuffer[meshID].d_pointer();

        //transform
        triangleInput[meshID].triangleArray.transformFormat = OPTIX_TRANSFORM_FORMAT_MATRIX_FLOAT12;
        triangleInput[meshID].triangleArray.preTransform = matrices[meshID].d_pointer();

        triangleInput[meshID].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangleInput[meshID].triangleArray.vertexStrideInBytes = sizeof(glm::vec3);
        triangleInput[meshID].triangleArray.numVertices = (int)mesh.vertecies.size();
        triangleInput[meshID].triangleArray.vertexBuffers = &d_vertices[meshID];

        triangleInput[meshID].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        triangleInput[meshID].triangleArray.indexStrideInBytes = sizeof(glm::ivec3);
        triangleInput[meshID].triangleArray.numIndexTriplets = (int)mesh.index.size();
        triangleInput[meshID].triangleArray.indexBuffer = d_indices[meshID];

        triangleInputFlags[meshID] = 0;

        // in this example we have one SBT entry, and no per-primitive
        // materials:
        triangleInput[meshID].triangleArray.flags = &triangleInputFlags[meshID];
        triangleInput[meshID].triangleArray.numSbtRecords = 1;
        triangleInput[meshID].triangleArray.sbtIndexOffsetBuffer = 0;
        triangleInput[meshID].triangleArray.sbtIndexOffsetSizeInBytes = 0;
        triangleInput[meshID].triangleArray.sbtIndexOffsetStrideInBytes = 0;
    }
    // ==================================================================
    // BLAS setup
    // ==================================================================

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE
        | OPTIX_BUILD_FLAG_ALLOW_COMPACTION
        ;
    accelOptions.motionOptions.numKeys = 1;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes blasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage
    (optixContext,
        &accelOptions,
        triangleInput.data(),
        (int)numMeshes,  // num_build_inputs
        &blasBufferSizes
    ));

    // ==================================================================
    // prepare compaction
    // ==================================================================

    CUDABuffer compactedSizeBuffer;
    compactedSizeBuffer.alloc(sizeof(uint64_t));

    OptixAccelEmitDesc emitDesc;
    emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = compactedSizeBuffer.d_pointer();

    // ==================================================================
    // execute build (main stage)
    // ==================================================================

    CUDABuffer tempBuffer;
    tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);

    CUDABuffer outputBuffer;
    outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);

    OPTIX_CHECK(optixAccelBuild(optixContext,
        /* stream */0,
        &accelOptions,
        triangleInput.data(),
        (int)numMeshes,
        tempBuffer.d_pointer(),
        tempBuffer.sizeInBytes,

        outputBuffer.d_pointer(),
        outputBuffer.sizeInBytes,

        &asHandle,

        &emitDesc, 1
    ));
    CUDA_SYNC_CHECK();

    // ==================================================================
    // perform compaction
    // ==================================================================
    uint64_t compactedSize;
    compactedSizeBuffer.download(&compactedSize, 1);

    asBuffer.alloc(compactedSize);
    OPTIX_CHECK(optixAccelCompact(optixContext,
        /*stream:*/0,
        asHandle,
        asBuffer.d_pointer(),
        asBuffer.sizeInBytes,
        &asHandle));
    CUDA_SYNC_CHECK();

    // ==================================================================
    // aaaaaand .... clean up
    // ==================================================================
    outputBuffer.free(); // << the UNcompacted, temporary output buffer
    tempBuffer.free();
    compactedSizeBuffer.free();

    return asHandle;
}

#pragma region SBT_Structures
/*! SBT record for a raygen program */
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // just a dummy value - later examples will use more interesting
    // data here
    void* data;
};

/*! SBT record for a miss program */
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // just a dummy value - later examples will use more interesting
    // data here
    void* data;
};

/*! SBT record for a hitgroup program */
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    MeshSBTData data;
};
#pragma endregion

// constructs the shader binding table
void OptixRenderer::BuildSBT() {
    // ------------------------------------------------------------------
    // build raygen records
    // ------------------------------------------------------------------
    std::vector<RaygenRecord> raygenRecords;
    for (int i = 0; i < raygenPGs.size(); i++) {
        RaygenRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(raygenPGs[i], &rec));
        rec.data = nullptr; /* for now ... */
        raygenRecords.push_back(rec);
    }
    raygenRecordsBuffer.alloc_and_upload(raygenRecords);
    sbt.raygenRecord = raygenRecordsBuffer.d_pointer();

    // ------------------------------------------------------------------
    // build miss records
    // ------------------------------------------------------------------
    std::vector<MissRecord> missRecords;
    for (int i = 0; i < missPGs.size(); i++) {
        MissRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(missPGs[i], &rec));
        rec.data = nullptr; /* for now ... */
        missRecords.push_back(rec);
    }
    missRecordsBuffer.alloc_and_upload(missRecords);
    sbt.missRecordBase = missRecordsBuffer.d_pointer();
    sbt.missRecordStrideInBytes = sizeof(MissRecord);
    sbt.missRecordCount = (int)missRecords.size();

    // ------------------------------------------------------------------
    // build hitgroup records
    // ------------------------------------------------------------------
    int numObjects = (int)model->meshes.size();
    std::vector<HitgroupRecord> hitgroupRecords;
    for (int meshID = 0; meshID < numObjects; meshID++) {
        for (int rayID = 0; rayID < RAY_TYPE_COUNT; rayID++) {
            auto mesh = model->meshes[meshID].get();

            HitgroupRecord rec;
            OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[rayID], &rec));
            rec.data.albedoColor = mesh->albedo;
            rec.data.roughness = mesh->roughness;
            rec.data.metallic = mesh->metallic;
            //AlbedoTexture
            rec.data.hasAlbedoTexture = mesh->HasAlbedoTex();
            if (mesh->HasAlbedoTex()) {
                rec.data.albedoTexture = textureObjects[mesh->albedoTex];
            }
            //NormalTexture
            rec.data.hasNormalTexture = mesh->HasAlbedoTex();
            if (mesh->HasNormalTex()) {
                rec.data.normalTexture = textureObjects[mesh->normalTex];
            }
            //MetalRoughTexture
            rec.data.hasMetalRoughTexture = mesh->HasAlbedoTex();
            if (mesh->HasMetalRoughTex()) {
                rec.data.metalRoughTexture = textureObjects[mesh->metalRoughTex];
            }

            rec.data.index = (glm::ivec3*)indexBuffer[meshID].d_pointer();
            rec.data.vertex = (glm::vec3*)vertexBuffer[meshID].d_pointer();
            rec.data.normal = (glm::vec3*)normalBuffer[meshID].d_pointer();
            rec.data.texcoord = (glm::vec2*)texcoordBuffer[meshID].d_pointer();
            hitgroupRecords.push_back(rec);
        }
    }
    hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
    sbt.hitgroupRecordBase = hitgroupRecordsBuffer.d_pointer();
    sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    sbt.hitgroupRecordCount = (int)hitgroupRecords.size();
}

void OptixRenderer::CreateTextures() {
    int numTextures = (int)model->textures.size();

    textureArrays.resize(numTextures);
    textureObjects.resize(numTextures);

    for (int textureID = 0; textureID < numTextures; textureID++) {
        auto texture = model->textures[textureID].get();

        cudaResourceDesc res_desc = {};

        cudaChannelFormatDesc channel_desc;
        int32_t width = texture->resolution.x;
        int32_t height = texture->resolution.y;
        int32_t numComponents = 4;
        int32_t pitch = width * numComponents * sizeof(uint8_t);
        channel_desc = cudaCreateChannelDesc<uchar4>();

        cudaArray_t& pixelArray = textureArrays[textureID];
        CUDA_CHECK(MallocArray(&pixelArray,
            &channel_desc,
            width, height));

        CUDA_CHECK(Memcpy2DToArray(pixelArray,
            /* offset */0, 0,
            texture->pixel,
            pitch, pitch, height,
            cudaMemcpyHostToDevice));

        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = pixelArray;

        cudaTextureDesc tex_desc = {};
        tex_desc.addressMode[0] = cudaAddressModeWrap;
        tex_desc.addressMode[1] = cudaAddressModeWrap;
        tex_desc.filterMode = cudaFilterModeLinear;
        tex_desc.readMode = cudaReadModeNormalizedFloat;
        tex_desc.normalizedCoords = 1;
        tex_desc.maxAnisotropy = 1;
        tex_desc.maxMipmapLevelClamp = 99;
        tex_desc.minMipmapLevelClamp = 0;
        tex_desc.mipmapFilterMode = cudaFilterModePoint;
        tex_desc.borderColor[0] = 1.0f;
        tex_desc.sRGB = 0;

        // Create texture object
        cudaTextureObject_t cuda_tex = 0;
        CUDA_CHECK(CreateTextureObject(&cuda_tex, &res_desc, &tex_desc, nullptr));
        textureObjects[textureID] = cuda_tex;
    }
}

#pragma endregion

// render one frame
void OptixRenderer::Render(uint32_t h_pixels[])
{
    // sanity check: make sure we launch only after first resize is
    // already done:
    if (launchParams.frame.size.x == 0) return;

    launchParamsBuffer.upload(&launchParams, 1);

    OPTIX_CHECK(optixLaunch(/*! pipeline we're launching launch: */
        pipeline, stream,
        /*! parameters and SBT */
        launchParamsBuffer.d_pointer(),
        launchParamsBuffer.sizeInBytes,
        &sbt,
        /*! dimensions of the launch: */
        launchParams.frame.size.x,
        launchParams.frame.size.y,
        1
    ));
    // sync - make sure the frame is rendered before we download and
    // display (obviously, for a high-performance application you
    // want to use streams and double-buffering, but for this simple
    // example, this will have to do)
    CUDA_SYNC_CHECK();

    colorBuffer.download(h_pixels, launchParams.frame.size.x * launchParams.frame.size.y);

    int test = 0;
}

void OptixRenderer::Resize(glm::ivec2& newSize) {
    // if window minimized
    if (newSize.x == 0 || newSize.y == 0) return;

    // resize our cuda frame buffer
    this->colorBuffer.resize(newSize.x * newSize.y * sizeof(uint32_t));

    // update the launch parameters that we'll pass to the optix
    // launch:
    this->launchParams.frame.size = newSize;
    this->launchParams.frame.colorBuffer = (uint32_t*)colorBuffer.d_pointer();
}

void OptixRenderer::SetCamera(Camera& camera) {
    const float aspect = launchParams.frame.size.x / float(launchParams.frame.size.y);

    launchParams.camera.position = camera.position;
    launchParams.camera.inverseViewMatrix = glm::inverse(camera.GetViewMatrix());
    launchParams.camera.inverseProjectionMatrix = glm::inverse(camera.GetProjectionMatrix(aspect));
}