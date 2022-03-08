
module EchoStateNetworks

using Random
using LinearAlgebra

abstract type ESNArch end
struct NoTeacherForcing <: ESNArch end
struct TeacherForcing <: ESNArch end
export EchoStateNetwork
export train!, predict!

mutable struct EchoStateNetwork{T<:AbstractFloat}
    Nr::Integer     # the number of reservoir neurons
    Ni::Integer     # the number of input dimension
    No::Integer     # the number of output dimension
    Nl::Integer     # the number of reservoir layers
    sparsity::T
    spectral_radius::Array{T}
    noise_level::T
    leaking_rate::Array{T}
    teacher_forcing::Bool
    input_scaling::T
    inner_scaling::T
    Wr # reservoir weights
    Wi  # input weights
    Wo::Matrix{T}   # readout weights
    Wf::Matrix{T}   # feedback weights
    state
    input
    output
    prediction
    bias_scaling::T
    rng::AbstractRNG
    function EchoStateNetwork{T}(;Ni::Integer=1,
        No::Integer=1,
        Nr::Integer=100,
        Nl::Integer=1,
        sparsity::AbstractFloat=0.95,
        spectral_radius::Array{T}=repeat(0.95,Nl),
        noise_level::AbstractFloat=0.001,
        leaking_rate::Array{T}=repeat(0.95,Nl),
        teacher_forcing::Bool=true,
        input_scaling::T = 1,
        inner_scaling::T = 1,
        bias_scaling::T =0.01,
        rng::AbstractRNG=MersenneTwister(rand(UInt32)),
       ) where T<:AbstractFloat
        esn = new()
        esn.Ni = Ni
        esn.No = No
        esn.Nr = Nr
        esn.Nl = Nl
        esn.sparsity = sparsity
        esn.spectral_radius = spectral_radius
        esn.teacher_forcing = teacher_forcing
        esn.noise_level = noise_level
        esn.leaking_rate = leaking_rate
        esn.input_scaling = input_scaling
        esn.inner_scaling = inner_scaling
        esn.rng = rng

            init_weights!(esn)

        return esn
    end
end

function init_weights!(esn::EchoStateNetwork{T}) where T<:AbstractFloat
    # init reservoir weight matrix
    esn.Wr = []

    for num_layer in 1:esn.Nl
        push!(esn.Wr, rand(esn.rng, T, esn.Nr, esn.Nr) .- T(0.5))
        esn.Wr[num_layer] = (1 - esn.leaking_rate[num_layer])*Matrix{Float64}(I, esn.Nr, esn.Nr)+ esn.leaking_rate[num_layer]*esn.Wr[num_layer]
        # reduce connections based on `sparsity`
        Ns = round(Int,length(esn.Wr[num_layer])*esn.sparsity)# 向上去数， 9.5取10的这种意思

        for i in [rand(esn.rng, 1:length(esn.Wr[num_layer])) for i in 1:Ns]# 中括号里面的意思是生成Wr个0-1的随机数，然后取前Ns个
            Temp_Wr = esn.Wr[num_layer]
            Temp_Wr[i] = zero(T)
            esn.Wr[num_layer] = Temp_Wr #这个意思是将其中sparsity里面的的数赋值为0，进行矩阵稀疏化
        end # end for i
        ## rescale the matrix to fit the `spectral radius`
        esn.Wr[num_layer] *= esn.spectral_radius[num_layer]/maximum(abs.(eigvals(esn.Wr[num_layer])))#所有元素自乘后面这一大串
        esn.Wr[num_layer] *= esn.inner_scaling
    end # end for num_layer


    # init input weight matrix
    esn.Wi =[]
    for num_layer in 1:esn.Nl
        if num_layer==1
            push!(esn.Wi,rand(esn.rng, T, esn.Nr, esn.Ni+1) .- T(0.5))
            esn.Wi[num_layer] *= esn.input_scaling
        else
            push!(esn.Wi,rand(esn.rng, T, esn.Nr, esn.Nr+1) .- T(0.5))
            esn.Wi[num_layer] *=esn.input_scaling
        end
    end
    # init feedback weight matrix 这里没有进行更改，因为用不到feedback
    esn.Wf = rand(esn.rng, T, esn.Nr, esn.No) .- T(0.5)

    return esn
end # end for function init_weights!

function activate(f::Function, x::Array)
    return map(f, x)
end

function update(esn::EchoStateNetwork{T}, state::Vector{T},
                input::Vector{T}, output::Vector{T}, arch::NoTeacherForcing, i::Integer) where T<:AbstractFloat
    return activate(tanh, esn.Wi[i]*[one(T)*esn.bias_scaling;input] + esn.Wr[i]*state)
    + esn.noise_level*(rand(esn.rng, T, esn.Nr)-T(0.5))
end

function update(esn::EchoStateNetwork{T}, state::Vector{T},
                input::Vector{T}, output::Vector{T}, arch::TeacherForcing, i::Integer) where T<:AbstractFloat
    return activate(tanh, esn.Wi[i]*[one(T)*esn.bias_scaling;input] + esn.Wr[i]*state + esn.Wf*output)
    + esn.noise_level*(rand(esn.rng, T, esn.Nr)-T(0.5))
end

function reservoir_states(esn::EchoStateNetwork{T}, inputs, outputs::Matrix{T}) where T<:AbstractFloat
    u = []
    Nd = size(inputs, 2)
    states = zeros(T, (esn.Nr, Nd+1,esn.Nl))
    for t in 2:(Nd+1)
        arch = esn.teacher_forcing ? TeacherForcing() : NoTeacherForcing()
        for i in 1:esn.Nl
            if i==1
                u = inputs[:,t-1,i]
                #从第一个输入开始向网络输入数据
            else
                u = states[:,t,i-1]
                #从第二个i-1的第t时刻数据作为>2层数的输入数据
            end
            states[:,t,i] = (one(T)-esn.leaking_rate[i])*states[:,t-1,i] + esn.leaking_rate[i]*update(esn, states[:,t-1,i], u, outputs[:,t-1], arch,i)
        end
    end
    return states
end

function train!(esn::EchoStateNetwork{T}, inputs::Matrix{T}, outputs::Matrix{T};
                                  discard::Integer=min(div(size(inputs,2),10), 100), reg::AbstractFloat=1e-8) where T<:AbstractFloat
    #计算每一层的state
    @assert(size(inputs, 1) == esn.Ni)
    @assert(size(outputs, 1) == esn.No)
    @assert(size(inputs, 2) == size(outputs, 2))
    Nd = size(inputs,2)
    inputs_for_reservior = []
    #第一维是维度数ni，第二维是时刻数t，第三维是层数nl
    push!(inputs_for_reservior,inputs)
    esn.state = reservoir_states(esn, inputs_for_reservior[1], outputs)
    for i in 1:(esn.Nl-1)
        push!(inputs_for_reservior,esn.state[:,:,i])
    end

    # extended system states
    X = zeros(esn.Nl*esn.Nr,Nd)
    for t in 1:size(inputs,2)
        for num_layer in 1:esn.Nl
            X[1+(num_layer-1)*esn.Nr:esn.Nr+(num_layer-1)*esn.Nr,t] = esn.state[:,t+1,num_layer]
        end
    end
    # discard initial transient
    Xe = vcat(X[:,discard+1:end],ones(1,Nd-discard))
    tXe = Xe'
    O = outputs[:,discard+1:end]

    # calc output weight matrix
    esn.Wo = O*tXe*pinv(Xe*tXe + reg*Matrix{T}(I, size(Xe,1), size(Xe,1)))

    # store last states
    return esn
end

function predict!(esn::EchoStateNetwork{T}, inputs::Matrix{T}, outputs::Matrix{T} ; cont::Bool=true) where T<:AbstractFloat
    Nd = size(inputs, 2)
    states = zeros(T, (esn.Nr, Nd+1,esn.Nl))
    if cont
        for i in esn.Nl
            states[:,1,i] = esn.state[:,end,i]
        end
    else
        inputs = hcat(zeros(T,esn.Ni), inputs)
        outputs = hcat(zeros(T,esn.No), outputs)
    end

    arch = esn.teacher_forcing ? TeacherForcing() : NoTeacherForcing()
    for t = 2:Nd+1
        for i in esn.Nl
            if i ==1
                u = inputs[:,t-1]
            else
                u = states[:,t,i-1]
            end
            states[:,t,i] = (one(T)-esn.leaking_rate[i])*states[:,t-1,i] + esn.leaking_rate[i]*update(esn, states[:,t-1,i], u, outputs[:,t-1], arch,i)
        end
    end

    X = zeros(esn.Nl*esn.Nr,Nd)
    for t in 1:size(inputs,2)
        for num_layer in 1:esn.Nl
            X[1+(num_layer-1)*esn.Nr:esn.Nr+(num_layer-1)*esn.Nr,t] = states[:,t+1,num_layer]
        end
    end
    Xi = vcat(X,ones(1,Nd).*esn.bias_scaling)
    esn.prediction = esn.Wo*Xi
    return esn.prediction
end

end # module
