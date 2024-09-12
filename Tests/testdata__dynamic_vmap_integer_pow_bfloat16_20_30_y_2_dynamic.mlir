"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x20x30xbf16>) -> tensor<?x20x30xbf16>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x20x30xbf16>):
    %0 = "stablehlo.multiply"(%arg1, %arg1) : (tensor<?x20x30xbf16>, tensor<?x20x30xbf16>) -> tensor<?x20x30xbf16>
    "func.return"(%0) : (tensor<?x20x30xbf16>) -> ()
  }) : () -> ()
}) : () -> ()

