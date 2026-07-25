from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

label_voice = LabelEncoder()
label_emotion = LabelEncoder()

voice_dataset["target_voice"] = label_voice.fit_transform(
    voice_dataset["target_voice"]
)

voice_dataset["target_emotion"] = label_emotion.fit_transform(
    voice_dataset["target_emotion"]
)

X = voice_dataset.drop(
    [
        "target_voice",
        "target_emotion"
    ],
    axis=1
)

y_voice = voice_dataset["target_voice"]
y_emotion = voice_dataset["target_emotion"]

X_train, X_test, yv_train, yv_test = train_test_split(
    X,
    y_voice,
    test_size=0.2,
    random_state=42
)