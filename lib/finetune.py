# In this example, we assume that the number of classes for the fine-tuning task is 10,
# and the loss function is categorical cross-entropy. You will need to adjust these values
# for your specific task. Also, note that we assume you already have a train_dataset and
# val_dataset that are formatted appropriately for your specific task.

from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

# Load the pre-trained model
model = load_model("path/to/pretrained/model.h5")

# Add an output layer for your specific task
num_classes = 10
model_output = Dense(num_classes, activation='softmax')(model.output)

# Create the fine-tuned model
finetuned_model = Model(inputs=model.input, outputs=model_output)

# Compile the fine-tuned model
learning_rate = 1e-4
optimizer = Adam(learning_rate)
loss_fn = 'categorical_crossentropy'
metrics = ['accuracy']
finetuned_model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

# Train the fine-tuned model
finetuned_model.fit(train_dataset, epochs=10, validation_data=val_dataset)