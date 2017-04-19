#!/usr/bin/env th
require('torch')
require('cutorch')
require('nn')
require('image')
require('paths')
require('cunn')
require('cudnn')

torch.setdefaulttensortype('torch.FloatTensor')

eye = 231              -- small net requires 231x231
label_nb = nil        -- label of 'bee'
mean = 118.380948/255  -- global mean used to train overfeat
std = 61.896913/255    -- global std used to train overfeat
intensity = 1          -- pixel intensity for gradient sign

path_model = 'model.net'
label = require('overfeat_label')
i = 0
base_path = "/home/neale/repos/Adversarial/models/vgg/data/originals"
diff_path = "/home/neale/repos/Adversarial/models/vgg/data/noise"
adv_path = "/home/neale/repos/Adversarial/models/vgg/data/adversarials"

model = torch.load(path_model)--:cuda()
model.modules[#model.modules] = nn.LogSoftMax()--:cuda()
loss = nn.ClassNLLCriterion()--:cuda()

print (model)
print (loss)
local function check_string(x)
    local ext = ".JPEG"
    if string.find(x, ext) then
	local res = string.format("%s/%s", base_path, x)
	return res
    else return nil
    
    end
end
local function gen_adv(im, model) 
    -- resize input/label
    begin = string.find(im, '_')+1
    final = string.find(im, 'J')-2
    img_num = string.sub(im, begin, final)
    im = string.format("%s/%s", base_path, im)
    local img = image.scale(image.load(im), '^'..eye)
    local tx = math.floor((img:size(3)-eye)/2) + 1
    local ly = math.floor((img:size(2)-eye)/2) + 1
    img = img[{{},{ly,ly+eye-1},{tx,tx+eye-1}}]
    img:add(-mean):div(std)
    
    local img_adv = require('adversarial-fast')(model, loss, img:clone(), label_nb, std, intensity)

    model.modules[#model.modules] = nn.SoftMax()
    --model:cuda()
    -- check prediction results
    local pred = model:forward(img)
    local val, idx = pred:max(pred:dim())
    print('==> original:', label[ idx[1] ], 'confidence:', val[1])

    local pred = model:forward(img_adv)
    local val, idx = pred:max(pred:dim())
    print('==> adversarial:', label[ idx[1] ], 'confidence:', val[1])

    local img_diff = torch.add(img, -img_adv)
    print('==> mean absolute diff between the original and adversarial images[min/max]:', torch.abs(img_diff):mean())

    print ("saving file to", adv_path)
    image.save(string.format("%s/adv_%s.JPEG", adv_path, img_num), img_adv:mul(std):add(mean):clamp(0,255))
    image.save(string.format("%s/diff_%s.JPEG", diff_path, img_num), img_diff)

end
for im in paths.files(base_path, check_string) do
	print (i)
    if pcall(gen_adv, im, model) then
	i = i + 1
    end
end
