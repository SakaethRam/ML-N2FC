def predict_voice(game_features):

    features = torch.tensor(game_features).float()

    prediction = model(features)

    voice = label_voice.inverse_transform(
        [prediction.argmax().item()]
    )[0]

    return {

        "voice": voice,

        "emotion": "auto",

        "pitch": "dynamic",

        "speed": "adaptive"

    }

context = get_live_context()

voice_profile = predict_voice(context)

elevenlabs.generate(
    text=current_transcript,
    voice=voice_profile["voice"]
)