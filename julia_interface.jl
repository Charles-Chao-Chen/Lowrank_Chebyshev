function bbfmm1d(f::Function, X::Array{Float64},Y::Array{Float64}, Rrank::Int64)
    U = zeros(size(X,1), Rrank)
    V = zeros(size(Y,1), Rrank)
    f_c = @cfunction($f, Cdouble, (Cdouble, Cdouble));
    ccall((:bbfmm1D,"build/liblowrank.so"), Cvoid,
            (Ptr{Cvoid}, Ref{Cdouble}, Ref{Cdouble}, Cdouble, Cdouble, Cdouble, Cdouble, Ref{Cdouble}, Ref{Cdouble}, Cint, Cint, Cint),
            f_c, X, Y, minimum(X), maximum(X) ,minimum(Y), maximum(Y) ,U,V, Rrank, length(X), length(Y))
    return U, V
end

function bbfmm2d(f::Function, X::Array{Float64,2},Y::Array{Float64,2}, Rrank::Int64)
    U = zeros(size(X,1), Rrank^2)
    V = zeros(size(Y,1), Rrank^2)
    f_c = @cfunction($f, Cdouble, (Cdouble, Cdouble, Cdouble, Cdouble));
    ccall((:bbfmm2D,"build/liblowrank.so"), Cvoid,
            (Ptr{Cvoid}, Ref{Cdouble}, Ref{Cdouble}, Ref{Cdouble}, Ref{Cdouble}, Cint, Cint, Cint),
            f_c, X[:], Y[:],U,V, Rrank, size(X,1), size(Y,1))
    return U, V
end

function test()
    f = (x,y)->1/(x^2+y^2+1)
    X = rand(10,1)
    Y = rand(10,1) .+ 5.0
    A = zeros(10,10)
    for i = 1:10
        for j = 1:10
            A[i,j] = f(X[i], Y[j])
        end
    end
    U, V = bbfmm1d(f, X, Y, 5)
    println("Error = ", maximum(abs.(A-U*V')))

    f = (x1, y1, x2, y2)->1/((x1-x2)^2+(y1-y2)^2+1)
    X = rand(10,2)
    Y = rand(10,2) .+ 5.0
    A = zeros(10,10)
    for i = 1:10
        for j = 1:10
            A[i,j] = f(X[i,1], X[i,2], Y[j,1], Y[j,2])
        end
    end
    U, V = bbfmm2d(f, X, Y, 5)
    println("Error = ", maximum(abs.(A-U*V')))
end