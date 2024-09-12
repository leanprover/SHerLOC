"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<9xf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %50:3 = "func.call"() <{callee = @inputs}> : () -> (tensor<9xf16>, tensor<9xf16>, tensor<9xf16>)
    %51 = "func.call"() <{callee = @expected}> : () -> tensor<9xf16>
    %52 = "stablehlo.convert"(%50#0) : (tensor<9xf16>) -> tensor<9xf32>
    %53 = "stablehlo.convert"(%50#1) : (tensor<9xf16>) -> tensor<9xf32>
    %54 = "stablehlo.convert"(%50#2) : (tensor<9xf16>) -> tensor<9xf32>
    %55 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %56 = "stablehlo.broadcast_in_dim"(%55) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<9xf32>
    %57 = "stablehlo.compare"(%52, %56) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LE>}> : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xi1>
    %58 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %59 = "stablehlo.broadcast_in_dim"(%58) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<9xf32>
    %60 = "stablehlo.compare"(%53, %59) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LE>}> : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xi1>
    %61 = "stablehlo.or"(%57, %60) : (tensor<9xi1>, tensor<9xi1>) -> tensor<9xi1>
    %62 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %63 = "stablehlo.broadcast_in_dim"(%62) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<9xf32>
    %64 = "stablehlo.compare"(%54, %63) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xi1>
    %65 = "stablehlo.or"(%61, %64) : (tensor<9xi1>, tensor<9xi1>) -> tensor<9xi1>
    %66 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %67 = "stablehlo.broadcast_in_dim"(%66) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<9xf32>
    %68 = "stablehlo.compare"(%54, %67) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xi1>
    %69 = "stablehlo.or"(%65, %68) : (tensor<9xi1>, tensor<9xi1>) -> tensor<9xi1>
    %70 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %71 = "stablehlo.broadcast_in_dim"(%70) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<9xf32>
    %72 = "stablehlo.add"(%52, %71) : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xf32>
    %73 = "stablehlo.add"(%52, %53) : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xf32>
    %74 = "stablehlo.constant"() <{value = dense<2.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %75 = "stablehlo.broadcast_in_dim"(%74) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<9xf32>
    %76 = "stablehlo.add"(%73, %75) : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xf32>
    %77 = "stablehlo.divide"(%72, %76) : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xf32>
    %78 = "stablehlo.compare"(%54, %77) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xi1>
    %79 = "stablehlo.select"(%78, %52, %53) : (tensor<9xi1>, tensor<9xf32>, tensor<9xf32>) -> tensor<9xf32>
    %80 = "stablehlo.select"(%78, %53, %52) : (tensor<9xi1>, tensor<9xf32>, tensor<9xf32>) -> tensor<9xf32>
    %81 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %82 = "stablehlo.broadcast_in_dim"(%81) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<9xf32>
    %83 = "stablehlo.subtract"(%82, %54) : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xf32>
    %84 = "stablehlo.select"(%78, %54, %83) : (tensor<9xi1>, tensor<9xf32>, tensor<9xf32>) -> tensor<9xf32>
    %85 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %86 = "stablehlo.broadcast_in_dim"(%85) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<9xi64>
    %87 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %88 = "stablehlo.broadcast_in_dim"(%87) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<9xi64>
    %89 = "stablehlo.compare"(%86, %88) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<9xi64>, tensor<9xi64>) -> tensor<9xi1>
    %90 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %91 = "stablehlo.broadcast_in_dim"(%90) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<9xf32>
    %92 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %93 = "stablehlo.broadcast_in_dim"(%92) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<9xf32>
    %94 = "stablehlo.select"(%89, %91, %93) : (tensor<9xi1>, tensor<9xf32>, tensor<9xf32>) -> tensor<9xf32>
    %95 = "stablehlo.abs"(%94) : (tensor<9xf32>) -> tensor<9xf32>
    %96 = "stablehlo.constant"() <{value = dense<5.96046448E-8> : tensor<f32>}> : () -> tensor<f32>
    %97 = "stablehlo.broadcast_in_dim"(%96) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<9xf32>
    %98 = "stablehlo.compare"(%95, %97) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xi1>
    %99 = "stablehlo.constant"() <{value = dense<5.96046448E-8> : tensor<f32>}> : () -> tensor<f32>
    %100 = "stablehlo.broadcast_in_dim"(%99) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<9xf32>
    %101 = "stablehlo.select"(%98, %100, %94) : (tensor<9xi1>, tensor<9xf32>, tensor<9xf32>) -> tensor<9xf32>
    %102 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %103 = "stablehlo.broadcast_in_dim"(%102) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<9xf32>
    %104 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
    %105 = "stablehlo.constant"() <{value = dense<true> : tensor<i1>}> : () -> tensor<i1>
    %106:8 = "stablehlo.while"(%79, %80, %84, %104, %105, %101, %103, %101) ({
    ^bb0(%arg21: tensor<9xf32>, %arg22: tensor<9xf32>, %arg23: tensor<9xf32>, %arg24: tensor<i64>, %arg25: tensor<i1>, %arg26: tensor<9xf32>, %arg27: tensor<9xf32>, %arg28: tensor<9xf32>):
      %215 = "stablehlo.constant"() <{value = dense<200> : tensor<i64>}> : () -> tensor<i64>
      %216 = "stablehlo.compare"(%arg24, %215) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i64>, tensor<i64>) -> tensor<i1>
      %217 = "stablehlo.and"(%216, %arg25) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      "stablehlo.return"(%217) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg11: tensor<9xf32>, %arg12: tensor<9xf32>, %arg13: tensor<9xf32>, %arg14: tensor<i64>, %arg15: tensor<i1>, %arg16: tensor<9xf32>, %arg17: tensor<9xf32>, %arg18: tensor<9xf32>):
      %131 = "stablehlo.broadcast_in_dim"(%arg14) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<9xi64>
      %132 = "stablehlo.constant"() <{value = dense<2> : tensor<i64>}> : () -> tensor<i64>
      %133 = "stablehlo.broadcast_in_dim"(%132) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<9xi64>
      %134 = "func.call"(%131, %133) <{callee = @remainder}> : (tensor<9xi64>, tensor<9xi64>) -> tensor<9xi64>
      %135 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
      %136 = "stablehlo.broadcast_in_dim"(%135) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<9xi64>
      %137 = "stablehlo.compare"(%134, %136) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<9xi64>, tensor<9xi64>) -> tensor<9xi1>
      %138 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
      %139 = "stablehlo.broadcast_in_dim"(%138) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<9xi64>
      %140 = "stablehlo.compare"(%131, %139) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<9xi64>, tensor<9xi64>) -> tensor<9xi1>
      %141 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
      %142 = "stablehlo.broadcast_in_dim"(%141) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<9xi64>
      %143 = "stablehlo.subtract"(%131, %142) : (tensor<9xi64>, tensor<9xi64>) -> tensor<9xi64>
      %144 = "stablehlo.constant"() <{value = dense<2> : tensor<i64>}> : () -> tensor<i64>
      %145 = "stablehlo.broadcast_in_dim"(%144) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<9xi64>
      %146 = "func.call"(%143, %145) <{callee = @floor_divide}> : (tensor<9xi64>, tensor<9xi64>) -> tensor<9xi64>
      %147 = "stablehlo.convert"(%146) : (tensor<9xi64>) -> tensor<9xf32>
      %148 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %149 = "stablehlo.broadcast_in_dim"(%148) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<9xf32>
      %150 = "stablehlo.constant"() <{value = dense<2.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %151 = "stablehlo.broadcast_in_dim"(%150) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<9xf32>
      %152 = "stablehlo.add"(%arg11, %147) : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xf32>
      %153 = "stablehlo.negate"(%152) : (tensor<9xf32>) -> tensor<9xf32>
      %154 = "stablehlo.add"(%arg11, %arg12) : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xf32>
      %155 = "stablehlo.add"(%154, %147) : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xf32>
      %156 = "stablehlo.multiply"(%153, %155) : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xf32>
      %157 = "stablehlo.multiply"(%156, %arg13) : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xf32>
      %158 = "stablehlo.multiply"(%151, %147) : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xf32>
      %159 = "stablehlo.add"(%arg11, %158) : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xf32>
      %160 = "stablehlo.multiply"(%151, %147) : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xf32>
      %161 = "stablehlo.add"(%arg11, %160) : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xf32>
      %162 = "stablehlo.add"(%161, %149) : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xf32>
      %163 = "stablehlo.multiply"(%159, %162) : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xf32>
      %164 = "stablehlo.divide"(%157, %163) : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xf32>
      %165 = "stablehlo.subtract"(%arg12, %147) : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xf32>
      %166 = "stablehlo.multiply"(%147, %165) : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xf32>
      %167 = "stablehlo.multiply"(%166, %arg13) : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xf32>
      %168 = "stablehlo.multiply"(%151, %147) : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xf32>
      %169 = "stablehlo.add"(%arg11, %168) : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xf32>
      %170 = "stablehlo.subtract"(%169, %149) : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xf32>
      %171 = "stablehlo.multiply"(%151, %147) : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xf32>
      %172 = "stablehlo.add"(%arg11, %171) : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xf32>
      %173 = "stablehlo.multiply"(%170, %172) : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xf32>
      %174 = "stablehlo.divide"(%167, %173) : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xf32>
      %175 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %176 = "stablehlo.broadcast_in_dim"(%175) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<9xf32>
      %177 = "stablehlo.select"(%137, %164, %174) : (tensor<9xi1>, tensor<9xf32>, tensor<9xf32>) -> tensor<9xf32>
      %178 = "stablehlo.select"(%140, %176, %177) : (tensor<9xi1>, tensor<9xf32>, tensor<9xf32>) -> tensor<9xf32>
      %179 = "stablehlo.broadcast_in_dim"(%arg14) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<9xi64>
      %180 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
      %181 = "stablehlo.broadcast_in_dim"(%180) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<9xi64>
      %182 = "stablehlo.compare"(%179, %181) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<9xi64>, tensor<9xi64>) -> tensor<9xi1>
      %183 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %184 = "stablehlo.broadcast_in_dim"(%183) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<9xf32>
      %185 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %186 = "stablehlo.broadcast_in_dim"(%185) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<9xf32>
      %187 = "stablehlo.select"(%182, %184, %186) : (tensor<9xi1>, tensor<9xf32>, tensor<9xf32>) -> tensor<9xf32>
      %188 = "stablehlo.divide"(%178, %arg16) : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xf32>
      %189 = "stablehlo.add"(%187, %188) : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xf32>
      %190 = "stablehlo.constant"() <{value = dense<5.96046448E-8> : tensor<f32>}> : () -> tensor<f32>
      %191 = "stablehlo.broadcast_in_dim"(%190) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<9xf32>
      %192 = "stablehlo.abs"(%189) : (tensor<9xf32>) -> tensor<9xf32>
      %193 = "stablehlo.compare"(%192, %191) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xi1>
      %194 = "stablehlo.select"(%193, %191, %189) : (tensor<9xi1>, tensor<9xf32>, tensor<9xf32>) -> tensor<9xf32>
      %195 = "stablehlo.multiply"(%178, %arg17) : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xf32>
      %196 = "stablehlo.add"(%187, %195) : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xf32>
      %197 = "stablehlo.abs"(%196) : (tensor<9xf32>) -> tensor<9xf32>
      %198 = "stablehlo.compare"(%197, %191) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xi1>
      %199 = "stablehlo.select"(%198, %191, %196) : (tensor<9xi1>, tensor<9xf32>, tensor<9xf32>) -> tensor<9xf32>
      %200 = "func.call"(%199) <{callee = @integer_pow}> : (tensor<9xf32>) -> tensor<9xf32>
      %201 = "stablehlo.multiply"(%194, %200) : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xf32>
      %202 = "stablehlo.multiply"(%arg18, %201) : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xf32>
      %203 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
      %204 = "stablehlo.add"(%arg14, %203) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      %205 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %206 = "stablehlo.broadcast_in_dim"(%205) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<9xf32>
      %207 = "stablehlo.subtract"(%201, %206) : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xf32>
      %208 = "stablehlo.abs"(%207) : (tensor<9xf32>) -> tensor<9xf32>
      %209 = "stablehlo.constant"() <{value = dense<5.96046448E-8> : tensor<f32>}> : () -> tensor<f32>
      %210 = "stablehlo.broadcast_in_dim"(%209) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<9xf32>
      %211 = "stablehlo.compare"(%208, %210) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GE>}> : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xi1>
      %212 = "stablehlo.constant"() <{value = dense<false> : tensor<i1>}> : () -> tensor<i1>
      %213 = "stablehlo.reduce"(%211, %212) <{dimensions = array<i64: 0>}> ({
      ^bb0(%arg19: tensor<i1>, %arg20: tensor<i1>):
        %214 = "stablehlo.or"(%arg19, %arg20) : (tensor<i1>, tensor<i1>) -> tensor<i1>
        "stablehlo.return"(%214) : (tensor<i1>) -> ()
      }) : (tensor<9xi1>, tensor<i1>) -> tensor<i1>
      "stablehlo.return"(%arg11, %arg12, %arg13, %204, %213, %194, %200, %202) : (tensor<9xf32>, tensor<9xf32>, tensor<9xf32>, tensor<i64>, tensor<i1>, tensor<9xf32>, tensor<9xf32>, tensor<9xf32>) -> ()
    }) : (tensor<9xf32>, tensor<9xf32>, tensor<9xf32>, tensor<i64>, tensor<i1>, tensor<9xf32>, tensor<9xf32>, tensor<9xf32>) -> (tensor<9xf32>, tensor<9xf32>, tensor<9xf32>, tensor<i64>, tensor<i1>, tensor<9xf32>, tensor<9xf32>, tensor<9xf32>)
    %107 = "chlo.lgamma"(%79) : (tensor<9xf32>) -> tensor<9xf32>
    %108 = "chlo.lgamma"(%80) : (tensor<9xf32>) -> tensor<9xf32>
    %109 = "stablehlo.add"(%107, %108) : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xf32>
    %110 = "stablehlo.add"(%79, %80) : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xf32>
    %111 = "chlo.lgamma"(%110) : (tensor<9xf32>) -> tensor<9xf32>
    %112 = "stablehlo.subtract"(%109, %111) : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xf32>
    %113 = "stablehlo.log"(%84) : (tensor<9xf32>) -> tensor<9xf32>
    %114 = "stablehlo.multiply"(%113, %79) : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xf32>
    %115 = "stablehlo.negate"(%84) : (tensor<9xf32>) -> tensor<9xf32>
    %116 = "stablehlo.log_plus_one"(%115) : (tensor<9xf32>) -> tensor<9xf32>
    %117 = "stablehlo.multiply"(%116, %80) : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xf32>
    %118 = "stablehlo.add"(%114, %117) : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xf32>
    %119 = "stablehlo.subtract"(%118, %112) : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xf32>
    %120 = "stablehlo.exponential"(%119) : (tensor<9xf32>) -> tensor<9xf32>
    %121 = "stablehlo.multiply"(%106#7, %120) : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xf32>
    %122 = "stablehlo.divide"(%121, %79) : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xf32>
    %123 = "stablehlo.constant"() <{value = dense<0x7FC00000> : tensor<f32>}> : () -> tensor<f32>
    %124 = "stablehlo.broadcast_in_dim"(%123) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<9xf32>
    %125 = "stablehlo.select"(%69, %124, %122) : (tensor<9xi1>, tensor<9xf32>, tensor<9xf32>) -> tensor<9xf32>
    %126 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %127 = "stablehlo.broadcast_in_dim"(%126) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<9xf32>
    %128 = "stablehlo.subtract"(%127, %125) : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xf32>
    %129 = "stablehlo.select"(%78, %125, %128) : (tensor<9xi1>, tensor<9xf32>, tensor<9xf32>) -> tensor<9xf32>
    %130 = "stablehlo.convert"(%129) : (tensor<9xf32>) -> tensor<9xf16>
    "stablehlo.custom_call"(%130, %51) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<9xf16>, tensor<9xf16>) -> ()
    "func.return"(%130) : (tensor<9xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<9xf16>, tensor<9xf16>, tensor<9xf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %47 = "stablehlo.constant"() <{value = dense<[-1.599610e+00, -1.400390e+00, -1.000000e+00, 0.000000e+00, 9.997550e-02, 3.000490e-01, 1.000000e+00, 1.400390e+00, 1.599610e+00]> : tensor<9xf16>}> : () -> tensor<9xf16>
    %48 = "stablehlo.constant"() <{value = dense<[-1.599610e+00, 1.400390e+00, 1.000000e+00, 0.000000e+00, 1.999510e-01, 9.997550e-02, 1.000000e+00, 1.400390e+00, -1.599610e+00]> : tensor<9xf16>}> : () -> tensor<9xf16>
    %49 = "stablehlo.constant"() <{value = dense<[1.000000e+00, -1.000000e+00, 2.000000e+00, 1.000000e+00, 3.000490e-01, 3.000490e-01, -1.000000e+00, 2.400390e+00, 1.599610e+00]> : tensor<9xf16>}> : () -> tensor<9xf16>
    "func.return"(%47, %48, %49) : (tensor<9xf16>, tensor<9xf16>, tensor<9xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<9xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %46 = "stablehlo.constant"() <{value = dense<[0x7E00, 0x7E00, 0x7E00, 0x7E00, 6.230460e-01, 1.945800e-01, 0x7E00, 0x7E00, 0x7E00]> : tensor<9xf16>}> : () -> tensor<9xf16>
    "func.return"(%46) : (tensor<9xf16>) -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], function_type = (tensor<9xi64>, tensor<9xi64>) -> tensor<9xi64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "remainder", sym_visibility = "private"}> ({
  ^bb0(%arg9: tensor<9xi64>, %arg10: tensor<9xi64>):
    %26 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %27 = "stablehlo.broadcast_in_dim"(%26) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<9xi64>
    %28 = "stablehlo.compare"(%arg10, %27) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<9xi64>, tensor<9xi64>) -> tensor<9xi1>
    %29 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
    %30 = "stablehlo.broadcast_in_dim"(%29) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<9xi64>
    %31 = "func.call"(%28, %30, %arg10) <{callee = @_where}> : (tensor<9xi1>, tensor<9xi64>, tensor<9xi64>) -> tensor<9xi64>
    %32 = "stablehlo.remainder"(%arg9, %31) : (tensor<9xi64>, tensor<9xi64>) -> tensor<9xi64>
    %33 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %34 = "stablehlo.broadcast_in_dim"(%33) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<9xi64>
    %35 = "stablehlo.compare"(%32, %34) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<9xi64>, tensor<9xi64>) -> tensor<9xi1>
    %36 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %37 = "stablehlo.broadcast_in_dim"(%36) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<9xi64>
    %38 = "stablehlo.compare"(%32, %37) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<9xi64>, tensor<9xi64>) -> tensor<9xi1>
    %39 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %40 = "stablehlo.broadcast_in_dim"(%39) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<9xi64>
    %41 = "stablehlo.compare"(%31, %40) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<9xi64>, tensor<9xi64>) -> tensor<9xi1>
    %42 = "stablehlo.compare"(%38, %41) <{compare_type = #stablehlo<comparison_type UNSIGNED>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<9xi1>, tensor<9xi1>) -> tensor<9xi1>
    %43 = "stablehlo.and"(%42, %35) : (tensor<9xi1>, tensor<9xi1>) -> tensor<9xi1>
    %44 = "stablehlo.add"(%32, %31) : (tensor<9xi64>, tensor<9xi64>) -> tensor<9xi64>
    %45 = "stablehlo.select"(%43, %44, %32) : (tensor<9xi1>, tensor<9xi64>, tensor<9xi64>) -> tensor<9xi64>
    "func.return"(%45) : (tensor<9xi64>) -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], function_type = (tensor<9xi1>, tensor<9xi64>, tensor<9xi64>) -> tensor<9xi64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "_where", sym_visibility = "private"}> ({
  ^bb0(%arg6: tensor<9xi1>, %arg7: tensor<9xi64>, %arg8: tensor<9xi64>):
    %25 = "stablehlo.select"(%arg6, %arg7, %arg8) : (tensor<9xi1>, tensor<9xi64>, tensor<9xi64>) -> tensor<9xi64>
    "func.return"(%25) : (tensor<9xi64>) -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], function_type = (tensor<9xi64>, tensor<9xi64>) -> tensor<9xi64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "floor_divide", sym_visibility = "private"}> ({
  ^bb0(%arg4: tensor<9xi64>, %arg5: tensor<9xi64>):
    %4 = "stablehlo.divide"(%arg4, %arg5) : (tensor<9xi64>, tensor<9xi64>) -> tensor<9xi64>
    %5 = "stablehlo.sign"(%arg4) : (tensor<9xi64>) -> tensor<9xi64>
    %6 = "stablehlo.sign"(%arg5) : (tensor<9xi64>) -> tensor<9xi64>
    %7 = "stablehlo.compare"(%5, %6) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<9xi64>, tensor<9xi64>) -> tensor<9xi1>
    %8 = "stablehlo.remainder"(%arg4, %arg5) : (tensor<9xi64>, tensor<9xi64>) -> tensor<9xi64>
    %9 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %10 = "stablehlo.broadcast_in_dim"(%9) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<9xi64>
    %11 = "stablehlo.compare"(%8, %10) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<9xi64>, tensor<9xi64>) -> tensor<9xi1>
    %12 = "stablehlo.constant"() <{value = dense<false> : tensor<i1>}> : () -> tensor<i1>
    %13 = "stablehlo.broadcast_in_dim"(%12) <{broadcast_dimensions = array<i64>}> : (tensor<i1>) -> tensor<9xi1>
    %14 = "stablehlo.compare"(%7, %13) <{compare_type = #stablehlo<comparison_type UNSIGNED>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<9xi1>, tensor<9xi1>) -> tensor<9xi1>
    %15 = "stablehlo.convert"(%14) : (tensor<9xi1>) -> tensor<9xi1>
    %16 = "stablehlo.constant"() <{value = dense<false> : tensor<i1>}> : () -> tensor<i1>
    %17 = "stablehlo.broadcast_in_dim"(%16) <{broadcast_dimensions = array<i64>}> : (tensor<i1>) -> tensor<9xi1>
    %18 = "stablehlo.compare"(%11, %17) <{compare_type = #stablehlo<comparison_type UNSIGNED>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<9xi1>, tensor<9xi1>) -> tensor<9xi1>
    %19 = "stablehlo.convert"(%18) : (tensor<9xi1>) -> tensor<9xi1>
    %20 = "stablehlo.and"(%15, %19) : (tensor<9xi1>, tensor<9xi1>) -> tensor<9xi1>
    %21 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
    %22 = "stablehlo.broadcast_in_dim"(%21) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<9xi64>
    %23 = "stablehlo.subtract"(%4, %22) : (tensor<9xi64>, tensor<9xi64>) -> tensor<9xi64>
    %24 = "func.call"(%20, %23, %4) <{callee = @_where_0}> : (tensor<9xi1>, tensor<9xi64>, tensor<9xi64>) -> tensor<9xi64>
    "func.return"(%24) : (tensor<9xi64>) -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], function_type = (tensor<9xi1>, tensor<9xi64>, tensor<9xi64>) -> tensor<9xi64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "_where_0", sym_visibility = "private"}> ({
  ^bb0(%arg1: tensor<9xi1>, %arg2: tensor<9xi64>, %arg3: tensor<9xi64>):
    %3 = "stablehlo.select"(%arg1, %arg2, %arg3) : (tensor<9xi1>, tensor<9xi64>, tensor<9xi64>) -> tensor<9xi64>
    "func.return"(%3) : (tensor<9xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<9xf32>) -> tensor<9xf32>, sym_name = "integer_pow", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<9xf32>):
    %0 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %1 = "stablehlo.broadcast_in_dim"(%0) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<9xf32>
    %2 = "stablehlo.divide"(%1, %arg0) : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xf32>
    "func.return"(%2) : (tensor<9xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

