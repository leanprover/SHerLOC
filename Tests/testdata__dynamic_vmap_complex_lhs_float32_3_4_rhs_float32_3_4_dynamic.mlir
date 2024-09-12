"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x3x4xf32>, tensor<?x3x4xf32>) -> tensor<?x3x4xcomplex<f32>>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x3x4xf32>, %arg2: tensor<?x3x4xf32>):
    %0 = "stablehlo.complex"(%arg1, %arg2) : (tensor<?x3x4xf32>, tensor<?x3x4xf32>) -> tensor<?x3x4xcomplex<f32>>
    "func.return"(%0) : (tensor<?x3x4xcomplex<f32>>) -> ()
  }) : () -> ()
}) : () -> ()

