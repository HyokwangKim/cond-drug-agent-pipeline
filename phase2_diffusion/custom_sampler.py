# phase2_diffusion/custom_sampler.py
import torch

def training_free_guided_generation(
    unet, 
    scheduler, 
    surrogate_model, 
    text_embeddings, 
    num_steps=50, 
    guidance_scale=15.0, 
    device="cpu"
):
    """
    사전 학습된 Diffusion 모델을 파인튜닝 없이 제어하는 Custom Sampling Loop.
    Tweedie's Formula와 Surrogate Model의 Gradient를 활용합니다.
    """
    print("\n🚀 [Phase 2] Training-Free Guidance 샘플링 시작...")
    
    # 1. 초기 노이즈(Latent Vector) 생성
    batch_size = text_embeddings.shape[0]
    in_channels = 4 # 임의의 채널 수 (실제 모델 구성에 따라 다름)
    latents = torch.randn((batch_size, in_channels, 64, 64), device=device)
    
    scheduler.set_timesteps(num_steps)
    
    # 2. 역방향 확산(Reverse Diffusion) 루프
    for i, t in enumerate(scheduler.timesteps):
        # ★ 핵심: 역전파를 위해 latents의 gradient 추적 활성화
        with torch.enable_grad():
            latents = latents.detach().requires_grad_(True)
            
            # 스케줄러에 맞게 입력 스케일링
            latent_model_input = scheduler.scale_model_input(latents, t)
            
            # UNet 노이즈 예측 (텍스트 조건부)
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings)
            
            # Tweedie's Formula: 예측된 노이즈를 바탕으로 깨끗한 원본(x0_hat) 역산
            alpha_prod_t = scheduler.alphas_cumprod[t]
            beta_prod_t = 1 - alpha_prod_t
            
            x0_hat = (latents - (beta_prod_t ** 0.5) * noise_pred) / (alpha_prod_t ** 0.5)
            
            # Surrogate Model을 통한 물성 예측 (Phase 1의 JSON 조건이 여기 반영됨)
            # 여기서는 예시로 대리 모델이 계산한 보상(Reward) 값을 가져온다고 가정
            reward = surrogate_model(x0_hat)
            
            # 보상을 최대화하기 위해 Loss는 음수로 설정
            loss = -1.0 * reward.sum()
            
            # x0_hat을 통과하여 현재 latents에 대한 그래디언트(방향) 계산
            grad = torch.autograd.grad(outputs=loss, inputs=latents)[0]
            
        # 외부 그래디언트 주입을 통한 노이즈 방향 수정 (Guidance Scale 적용)
        guided_noise_pred = noise_pred - (guidance_scale * torch.sqrt(beta_prod_t) * grad)
        
        # 수정된 노이즈로 다음 스텝 진행 (x_t -> x_t-1)
        # torch.no_grad() 환경에서 업데이트를 수행해야 메모리 누수가 발생하지 않습니다.
        with torch.no_grad():
            latents = scheduler.step(guided_noise_pred, t, latents).prev_sample
            
        if i % 10 == 0:
            print(f"  -> Step {i}/{num_steps} 완료 (Gradient 주입됨)")

    print("✅ 최적화된 분자 잠재 벡터(Latent Vector) 생성 완료!")
    return latents