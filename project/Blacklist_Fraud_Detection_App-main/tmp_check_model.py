import joblib, numpy as np, torch, os
from XBNet.training_utils import predict, predict_proba

model_path = os.path.join('instance','v2.pkl')
print('Loading model:', model_path)
# Make sure the module used when pickling is available (models_xb)
try:
    import importlib, sys
    mod = importlib.import_module('XBNet.models_xb')
    sys.modules['models_xb'] = mod
except Exception:
    pass

model = joblib.load(model_path)
print('Model loaded. Model name:', getattr(model,'name',None),'labels:',getattr(model,'labels',None))
# Create a synthetic input matching model.X shape if available
if hasattr(model,'X') and model.X is not None:
    X = np.array(model.X)
    dim = X.shape[1]
else:
    dim = 68
print('Input dim used for test:', dim)
# fabricate a varied input
test = np.linspace(0.1, 10.0, dim)
print('Test input (first 10):', test[:10])
# Print model training data stats if available
try:
    if hasattr(model, 'X') and model.X is not None:
        mX = np.array(model.X).astype(float)
        m_mean = np.nanmean(mX, axis=0)
        m_std = np.nanstd(mX, axis=0)
        print('Model X mean (first 10):', m_mean[:10])
        print('Model X std  (first 10):', m_std[:10])
    else:
        print('Model has no X attribute for stats')
except Exception as e:
    print('Could not compute model X stats:', e)

# Show scaled input using model stats if available
try:
    if hasattr(model, 'X') and model.X is not None:
        if mX.shape[1] == test.shape[0]:
            m_std[m_std == 0] = 1.0
            scaled = (test - m_mean) / m_std
            print('Scaled test input (first 10):', scaled[:10])
        else:
            print('Model X shape does not match test dim; skipping scaling')
except Exception as e:
    print('Error scaling test input:', e)
# run predict and predict_proba
pred = predict(model, test)
proba = predict_proba(model, test)
print('predict ->', pred)
print('predict_proba ->', proba)
# also get raw model output (logits)
model.eval()
with torch.no_grad():
    import torch
    t = torch.from_numpy(test.astype('float32')).unsqueeze(0)
    out = model(t, train=False)
    print('raw output (torch):', out)
    try:
        print('sigmoid(raw):', torch.sigmoid(out))
        print('softmax(raw):', torch.nn.functional.softmax(out, dim=1))
    except Exception as e:
        print('Could not apply activation to raw output:', e)
