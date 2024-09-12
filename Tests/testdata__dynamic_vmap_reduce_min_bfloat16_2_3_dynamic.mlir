"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x2x3xbf16>) -> tensor<?x3xbf16>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x2x3xbf16>):
    %0 = "stablehlo.constant"() <{value = dense<0x7F80> : tensor<bf16>}> : () -> tensor<bf16>
    %1 = "stablehlo.reduce"(%arg1, %0) <{dimensions = array<i64: 1>}> ({
    ^bb0(%arg2: tensor<bf16>, %arg3: tensor<bf16>):
      %2 = "stablehlo.minimum"(%arg2, %arg3) : (tensor<bf16>, tensor<bf16>) -> tensor<bf16>
      "stablehlo.return"(%2) : (tensor<bf16>) -> ()
    }) : (tensor<?x2x3xbf16>, tensor<bf16>) -> tensor<?x3xbf16>
    "func.return"(%1) : (tensor<?x3xbf16>) -> ()
  }) : () -> ()
}) : () -> ()

