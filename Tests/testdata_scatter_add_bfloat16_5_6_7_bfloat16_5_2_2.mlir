"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x6x7xbf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[[0, 1], [2, 3]], [[4, 0], [1, 2]]]> : tensor<2x2x2xi64>}> : () -> tensor<2x2x2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<5x6x7xbf16>, tensor<5x2x2xbf16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<5x6x7xbf16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1, 2], scatter_dims_to_operand_dims = [1, 2], index_vector_dim = 2>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %7 = "stablehlo.add"(%arg0, %arg1) : (tensor<bf16>, tensor<bf16>) -> tensor<bf16>
      "stablehlo.return"(%7) : (tensor<bf16>) -> ()
    }) : (tensor<5x6x7xbf16>, tensor<2x2x2xi64>, tensor<5x2x2xbf16>) -> tensor<5x6x7xbf16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<5x6x7xbf16>, tensor<5x6x7xbf16>) -> ()
    "func.return"(%6) : (tensor<5x6x7xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<5x6x7xbf16>, tensor<5x2x2xbf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x23C0CBBF683F52C035C0D5BF03C0464076BF14C07B3FE8BF0BC02EBF93BE6CBF32C0423E4A402FC0B5BFA03FB6BF49C0A4C0F5BF4DC048C009C0073F953E024063BF963F1C40503FA33F77BFDA3F1A3EB2BE3440D840D7BE4740DFBF3C3FC93F0CBF6F40AFBDCBBF7AC028408F40B3BE44C07D400040CB3F474038BFBE3F513F1940633FBABFFE3E4FC07CC07D3FBE3F98409B3E66C0BE3CBFBFC63F8DC04F3F83BF08C00BC0E4BE83C0ACC0DDC0A64091400A40BCBEB3BE5EC0EFC07E40DD40C53F0240463FB44003BF2B3E9DBF1C3E383F7D40113FFBBF91C092BF8CC00D40A43EE23ECABF413F39BE6640A5BF66BFBB3F0AC0FC3F603F304006C0B53E00BFC33F5A40133F7FBFE4C0433FBD3F9DC01C3E803F383D47403AC0613FCA3F83C0C3BFADBFD1BF7ABF4BC01CC08BBD37BF53C0783F19409C404D4052C00CC19EC0BA3F7E3E77C04840AF3E72400FBFB13E6EC03ABE434039C0363E113F2F408E3FBEBF9ABFD1403C40493F07406E3F62BC223FAEC03EC01EBFAE404C3F29C0BAC011C048C056C05F3F8D3F7B40A740793FDB3FF7BF804070C0973DF63F4C4031BF0C3F00BF"> : tensor<5x6x7xbf16>}> : () -> tensor<5x6x7xbf16>
    %2 = "stablehlo.constant"() <{value = dense<[[[-1.351560e+00, -9.765620e-01], [2.046880e+00, 1.156250e+00]], [[-1.179690e+00, -5.976560e-01], [2.156250e+00, -8.203130e-02]], [[3.250000e+00, -4.437500e+00], [-6.953130e-01, 3.125000e+00]], [[-3.687500e+00, 7.148430e-01], [-7.304680e-01, 2.234380e+00]], [[-1.728520e-01, -1.171880e+00], [-3.250000e+00, 1.312500e+00]]]> : tensor<5x2x2xbf16>}> : () -> tensor<5x2x2xbf16>
    "func.return"(%1, %2) : (tensor<5x6x7xbf16>, tensor<5x2x2xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x6x7xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x23C03CC0683F52C035C0D5BF03C0464076BF94BF7B3FE8BF0BC02EBF93BE6CBF32C04ABF4A402FC0B5BFA03FB6BF49C0A4C0F5BF4DC048C0C0BD073F953E024063BF963F1C40503FA33F77BFDA3F1A3EB2BE3440D840CDBF4740DFBF3C3FC93F0CBF6F40AFBDD6BF7AC028408F40B3BE44C07D4000407D3F474038BFBE3F513F1940633FBABFFE3E4FC07CC04940BE3F98409B3E66C0BE3CBFBFC63F8DC04F3F83BF08C00BC0E4BE83C008C0DDC0A64091400A40BCBEB3BE5EC08BC07E40DD40C53F0240463FB44003BF89C09DBF1C3E383F7D40113FFBBF91C092BF8CC00D40C0BEE23ECABF413F39BE6640A5BF66BFBB3F0AC0FC3F603F304006C0B53E86C0C33F5A40133F7FBFE4C0433FBD3F2BC01C3E803F383D47403AC0613FCA3F58C0C3BFADBFD1BF7ABF4BC01CC08BBD37BF53C0783FD43F9C404D4052C00CC19EC0BA3F7E3E77C04840AF3E72400FBFB13E6EC0B6BE434039C0363E113F2F408E3FBEBFE03DD1403C40493F07406E3F62BC223FD4C03EC01EBFAE404C3F29C0BAC011C048C056C05F3F0AC07B40A740793FDB3FF7BF804070C0973DF63F4C4031BF0C3F00BF"> : tensor<5x6x7xbf16>}> : () -> tensor<5x6x7xbf16>
    "func.return"(%0) : (tensor<5x6x7xbf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

